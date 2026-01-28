#!/usr/bin/env python3
"""
Nano ML Training Script - Training from Scratch
================================================

This script trains a ~1B parameter language model FROM SCRATCH on multilingual
(German/English) and code generation datasets with automatic checkpointing
to Hugging Face Hub.

Features:
- Model initialization from scratch (random weights)
- LLaMA-style architecture (~1B parameters)
- Automatic dataset downloading and caching
- Step-based checkpointing with Hub uploads
- Resumable training from checkpoints
- Comprehensive logging

Usage:
    python train.py [--resume]

Environment Variables:
    HF_TOKEN: Your Hugging Face authentication token
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator, Tuple
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, Dataset, concatenate_datasets, IterableDataset as HFIterableDataset
from huggingface_hub import HfApi, login, upload_folder
from tqdm import tqdm

from config import (
    get_config,
    get_config_for_vram,
    print_config,
    Config,
    GPUMemoryMode,
    print_memory_modes_summary,
)

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(config: Config) -> logging.Logger:
    """Configure logging for the training run."""
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pretraining_{timestamp}.log"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.logging.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger("nano-pretraining")
    logger.info(f"Logging to {log_file}")

    return logger


# =============================================================================
# HUGGING FACE HUB CALLBACK
# =============================================================================

class HubUploadCallback(TrainerCallback):
    """
    Custom callback to upload checkpoints to Hugging Face Hub.

    This callback handles:
    - Creating the repository if it doesn't exist
    - Uploading model checkpoints at specified intervals
    - Uploading training metrics and logs
    - Retry logic for failed uploads
    """

    def __init__(
        self,
        config: Config,
        logger: logging.Logger,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.config = config
        self.logger = logger
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api = HfApi(token=config.huggingface.token)
        self.repo_created = False
        self.upload_history: List[Dict[str, Any]] = []
        self.last_upload_step = 0

    def _ensure_repo_exists(self):
        """Create the Hub repository if it doesn't exist."""
        if self.repo_created:
            return

        try:
            self.api.create_repo(
                repo_id=self.config.huggingface.repo_id,
                repo_type="model",
                private=self.config.huggingface.private,
                exist_ok=True,
            )
            self.repo_created = True
            self.logger.info(f"Repository '{self.config.huggingface.repo_id}' is ready.")
        except Exception as e:
            self.logger.error(f"Failed to create repository: {e}")
            raise

    def _upload_with_retry(
        self,
        folder_path: str,
        commit_message: str,
    ) -> bool:
        """Upload folder to Hub with retry logic."""
        self._ensure_repo_exists()

        for attempt in range(self.max_retries):
            try:
                self.logger.info(
                    f"Uploading to Hub (attempt {attempt + 1}/{self.max_retries})..."
                )

                upload_folder(
                    repo_id=self.config.huggingface.repo_id,
                    folder_path=folder_path,
                    commit_message=commit_message,
                    token=self.config.huggingface.token,
                    ignore_patterns=["optimizer.pt", "scheduler.pt", "rng_state.pth"],
                )

                self.logger.info("Upload successful!")
                return True

            except Exception as e:
                self.logger.warning(f"Upload attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)

        self.logger.error("All upload attempts failed.")
        return False

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called when a checkpoint is saved - upload to Hub."""
        if not self.config.huggingface.push_to_hub:
            return

        # Get current step and loss
        current_step = state.global_step
        current_loss = state.log_history[-1].get("loss", 0.0) if state.log_history else 0.0

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"CHECKPOINT AT STEP {current_step} - Starting Hub upload...")
        self.logger.info(f"{'='*60}")

        # Find the checkpoint directory
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{current_step}"

        if not checkpoint_dir.exists():
            self.logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return

        # Save training state info
        state_info = {
            "step": current_step,
            "epoch": state.epoch,
            "loss": current_loss,
            "timestamp": datetime.now().isoformat(),
            "model_config": {
                "model_name": self.config.model.model_name,
                "hidden_size": self.config.model.hidden_size,
                "num_layers": self.config.model.num_hidden_layers,
                "num_params": self.config.model.get_num_parameters(),
            },
        }

        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state_info, f, indent=2)

        # Format commit message
        commit_message = f"Checkpoint at step {current_step} - Loss: {current_loss:.4f}"

        # Upload to Hub
        success = self._upload_with_retry(
            folder_path=str(checkpoint_dir),
            commit_message=commit_message,
        )

        # Track upload history
        self.upload_history.append({
            "step": current_step,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "loss": current_loss,
        })

        self.last_upload_step = current_step

        if success:
            self.logger.info(f"Successfully uploaded checkpoint at step {current_step}")
        else:
            self.logger.error(f"Failed to upload checkpoint at step {current_step}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of training - upload final model."""
        if not self.config.huggingface.push_to_hub:
            return

        self.logger.info("\n" + "=" * 60)
        self.logger.info("TRAINING COMPLETED - Uploading final model...")
        self.logger.info("=" * 60)

        # Save upload history
        history_file = Path(args.output_dir) / "upload_history.json"
        with open(history_file, "w") as f:
            json.dump(self.upload_history, f, indent=2)

        # Upload final model
        final_dir = Path(args.output_dir) / "final_model"
        if final_dir.exists():
            success = self._upload_with_retry(
                folder_path=str(final_dir),
                commit_message=f"Final model - Training completed at step {state.global_step}",
            )

            if success:
                self.logger.info("Final model uploaded successfully!")
            else:
                self.logger.error("Failed to upload final model.")


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

def create_model_from_scratch(config: Config, logger: logging.Logger) -> LlamaForCausalLM:
    """
    Create a LLaMA-style model from scratch with random initialization.

    This creates a model with approximately 1B parameters using the
    configuration specified in config.model.
    """
    logger.info("Creating model from scratch...")
    logger.info(f"  Model name: {config.model.model_name}")
    logger.info(f"  Hidden size: {config.model.hidden_size}")
    logger.info(f"  Num layers: {config.model.num_hidden_layers}")
    logger.info(f"  Num attention heads: {config.model.num_attention_heads}")
    logger.info(f"  Intermediate size: {config.model.intermediate_size}")

    # Create LLaMA configuration
    model_config = LlamaConfig(
        vocab_size=config.model.vocab_size,
        hidden_size=config.model.hidden_size,
        intermediate_size=config.model.intermediate_size,
        num_hidden_layers=config.model.num_hidden_layers,
        num_attention_heads=config.model.num_attention_heads,
        num_key_value_heads=config.model.num_key_value_heads,
        max_position_embeddings=config.model.max_position_embeddings,
        rope_theta=config.model.rope_theta,
        rms_norm_eps=config.model.rms_norm_eps,
        attention_dropout=config.model.attention_dropout,
        hidden_act=config.model.hidden_act,
        tie_word_embeddings=config.model.tie_word_embeddings,
        initializer_range=config.model.initializer_range,
        attention_bias=config.model.attention_bias,
        mlp_bias=config.model.mlp_bias,
        bos_token_id=config.model.bos_token_id,
        eos_token_id=config.model.eos_token_id,
        pad_token_id=config.model.pad_token_id,
    )

    # Determine dtype
    dtype = getattr(torch, config.model.torch_dtype)

    # Create model with random initialization
    logger.info("Initializing model weights randomly...")

    # Check for Flash Attention 2
    attn_implementation = None
    if config.model.use_flash_attention_2:
        try:
            attn_implementation = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        except Exception:
            logger.warning("Flash Attention 2 not available, using default attention")
            attn_implementation = None

    model = LlamaForCausalLM(model_config)

    # Convert to appropriate dtype
    model = model.to(dtype)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model created with {num_params:,} parameters ({num_params/1e9:.2f}B)")
    logger.info(f"Trainable parameters: {num_trainable:,}")

    return model


def load_tokenizer(config: Config, logger: logging.Logger):
    """Load tokenizer from pretrained (we don't train tokenizer from scratch)."""
    logger.info(f"Loading tokenizer: {config.tokenizer.tokenizer_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer.tokenizer_name,
        token=config.huggingface.token,
        trust_remote_code=True,
    )

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set padding side
    tokenizer.padding_side = config.tokenizer.padding_side
    tokenizer.truncation_side = config.tokenizer.truncation_side

    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    return tokenizer


# =============================================================================
# DATASET LOADING AND PREPROCESSING
# =============================================================================

def format_text(example: Dict[str, Any], ds_config: dict) -> str:
    """
    Extract and format text from example based on dataset configuration.

    Handles multiple formats:
    - Chat messages (messages field)
    - Reasoning/CoT (question + step-by-step answer)
    - Simple text fields
    """
    text_field = ds_config.get("text_field", "text")
    answer_field = ds_config.get("answer_field")
    system_field = ds_config.get("system_field")
    format_type = ds_config.get("format", "text")

    # Handle chat/messages format
    if text_field == "messages" and "messages" in example:
        messages = example["messages"]
        if isinstance(messages, list):
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|{role}|>\n{content}")
            return "\n".join(parts) + "\n<|end|>"

    # Handle reasoning/chain-of-thought format
    if format_type == "reasoning" and answer_field:
        question = example.get(text_field, "")
        answer = example.get(answer_field, "")

        if not question or not answer:
            return ""

        # Build reasoning format with clear structure
        parts = []

        # Add system prompt if available
        if system_field and system_field in example:
            system = example[system_field]
            if system:
                parts.append(f"<|system|>\n{system}")

        # Add the question/problem
        parts.append(f"<|user|>\n{question}")

        # Add the reasoning/answer with thinking markers
        # This encourages the model to learn step-by-step reasoning
        if "####" in str(answer):
            # GSM8K style: reasoning #### final_answer
            reasoning_part, final_answer = str(answer).rsplit("####", 1)
            parts.append(f"<|assistant|>\n<think>\n{reasoning_part.strip()}\n</think>\n\nThe answer is: {final_answer.strip()}")
        elif "\\boxed{" in str(answer):
            # LaTeX boxed answer format
            parts.append(f"<|assistant|>\n<think>\n{answer}\n</think>")
        else:
            # General reasoning response
            parts.append(f"<|assistant|>\n{answer}")

        parts.append("<|end|>")
        return "\n".join(parts)

    # Handle simple text field
    if text_field in example:
        text = str(example[text_field])

        # If there's an answer field but not reasoning format, combine them
        if answer_field and answer_field in example:
            answer = str(example[answer_field])
            return f"<|user|>\n{text}\n<|assistant|>\n{answer}\n<|end|>"

        return text

    return ""


def load_streaming_dataset(
    ds_config: dict,
    tokenizer,
    max_seq_length: int,
    cache_dir: str,
    logger: logging.Logger,
) -> Iterator[Dict[str, Any]]:
    """Load a streaming dataset and yield tokenized examples."""
    ds_name = ds_config["name"]
    ds_subset = ds_config.get("config")
    ds_split = ds_config.get("split", "train")
    max_samples = ds_config.get("max_samples", float("inf"))

    logger.info(f"Loading streaming dataset: {ds_name}")

    try:
        dataset = load_dataset(
            ds_name,
            ds_subset,
            split=ds_split,
            streaming=True,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        count = 0
        for example in dataset:
            if count >= max_samples:
                break

            text = format_text(example, ds_config)
            if not text or len(text.strip()) < 10:
                continue

            # Tokenize
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                return_tensors=None,
            )

            if len(tokens["input_ids"]) < 10:
                continue

            yield {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
            }
            count += 1

    except Exception as e:
        logger.warning(f"Failed to load dataset {ds_name}: {e}")


def _load_single_dataset(
    ds_config: dict,
    cache_dir: str,
    logger: logging.Logger,
    progress_lock: threading.Lock,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Load a single dataset. This function is designed to run in parallel.

    Args:
        ds_config: Dataset configuration dictionary
        cache_dir: Directory to cache downloaded datasets
        logger: Logger instance
        progress_lock: Lock for thread-safe logging

    Returns:
        Tuple of (dataset_name, list of examples)
    """
    ds_name = ds_config["name"]
    ds_subset = ds_config.get("config")
    ds_split = ds_config.get("split", "train")
    streaming = ds_config.get("streaming", False)
    max_samples = ds_config.get("max_samples", 100000)

    with progress_lock:
        logger.info(f"Starting download: {ds_name} (config: {ds_subset}, split: {ds_split})")

    examples = []

    try:
        if streaming:
            # Load streaming dataset
            dataset = load_dataset(
                ds_name,
                ds_subset,
                split=ds_split,
                streaming=True,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

            # Collect samples from streaming dataset
            count = 0
            for example in dataset:
                if count >= max_samples:
                    break

                text = format_text(example, ds_config)
                if text and len(text.strip()) >= 10:
                    examples.append({"text": text})
                    count += 1

                # Log progress every 10000 samples
                if count % 10000 == 0:
                    with progress_lock:
                        logger.info(f"  {ds_name}: loaded {count}/{max_samples} samples")
        else:
            # Load regular dataset with parallel download
            dataset = load_dataset(
                ds_name,
                ds_subset,
                split=ds_split,
                cache_dir=cache_dir,
                trust_remote_code=True,
                num_proc=4,  # Parallel processing for non-streaming datasets
            )

            # Process samples
            for i, example in enumerate(dataset):
                if i >= max_samples:
                    break

                text = format_text(example, ds_config)
                if text and len(text.strip()) >= 10:
                    examples.append({"text": text})

        with progress_lock:
            logger.info(f"Completed: {ds_name} - loaded {len(examples)} examples")

    except Exception as e:
        with progress_lock:
            logger.warning(f"Failed to load dataset {ds_name}: {e}")

    return ds_name, examples


def load_and_prepare_datasets(
    config: Config,
    tokenizer,
    logger: logging.Logger,
    max_workers: int = 4,
) -> Dataset:
    """
    Load, preprocess, and combine all datasets using parallel downloading.

    This function downloads multiple datasets concurrently to significantly
    reduce the total time required for data preparation.

    Args:
        config: Configuration object
        tokenizer: Tokenizer instance (unused but kept for API compatibility)
        logger: Logger instance
        max_workers: Maximum number of parallel download workers (default: 4)

    Returns:
        Combined Dataset ready for tokenization
    """
    logger.info("=" * 60)
    logger.info("PARALLEL DATASET LOADING")
    logger.info("=" * 60)
    logger.info(f"Loading {len(config.dataset.datasets)} datasets with {max_workers} parallel workers...")

    dataset_configs = config.dataset.datasets
    all_examples = []
    progress_lock = threading.Lock()

    start_time = time.time()

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all dataset loading tasks
        future_to_ds = {
            executor.submit(
                _load_single_dataset,
                ds_config,
                config.dataset.cache_dir,
                logger,
                progress_lock,
            ): ds_config["name"]
            for ds_config in dataset_configs
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_ds):
            ds_name = future_to_ds[future]
            completed += 1

            try:
                _, examples = future.result()
                if examples:
                    all_examples.extend(examples)
                    logger.info(
                        f"[{completed}/{len(dataset_configs)}] Added {len(examples)} examples from {ds_name}"
                    )
                else:
                    logger.warning(f"[{completed}/{len(dataset_configs)}] No examples from {ds_name}")
            except Exception as e:
                logger.error(f"[{completed}/{len(dataset_configs)}] Error processing {ds_name}: {e}")

    elapsed_time = time.time() - start_time

    if not all_examples:
        raise ValueError("No datasets were loaded successfully!")

    # Create combined dataset
    logger.info(f"Combining {len(all_examples)} total examples...")
    combined_dataset = Dataset.from_list(all_examples)

    # Shuffle
    combined_dataset = combined_dataset.shuffle(seed=config.dataset.seed)

    logger.info("=" * 60)
    logger.info(f"DATASET LOADING COMPLETED")
    logger.info(f"  Total examples: {len(combined_dataset)}")
    logger.info(f"  Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    logger.info(f"  Throughput: {len(combined_dataset)/elapsed_time:.0f} examples/second")
    logger.info("=" * 60)

    return combined_dataset


def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    config: Config,
    logger: logging.Logger,
) -> Dataset:
    """Tokenize the dataset for training."""
    logger.info("Tokenizing dataset...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.dataset.max_seq_length,
            padding=False,
            return_tensors=None,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
        num_proc=config.dataset.num_workers,
    )

    # Filter out very short sequences
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) >= 32
    )

    logger.info(f"Tokenized dataset size: {len(tokenized_dataset)}")

    return tokenized_dataset


# =============================================================================
# TRAINING
# =============================================================================

def get_training_arguments(config: Config) -> TrainingArguments:
    """Create training arguments from configuration."""
    # Determine optimizer based on memory configuration
    optim = config.training.optim
    if config.memory and config.memory.use_8bit_optimizer:
        # Use 8-bit AdamW from bitsandbytes for memory efficiency
        optim = "adamw_bnb_8bit"

    # Build training arguments
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        lr_scheduler_type=config.training.lr_scheduler_type,
        optim=optim,
        adam_beta1=config.training.adam_beta1,
        adam_beta2=config.training.adam_beta2,
        adam_epsilon=config.training.adam_epsilon,
        max_grad_norm=config.training.max_grad_norm,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        tf32=config.memory.tf32 if config.memory else True,
        gradient_checkpointing=config.training.gradient_checkpointing,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        seed=config.training.seed,
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataloader_pin_memory=config.training.dataloader_pin_memory,
        dataloader_drop_last=config.training.dataloader_drop_last,
        dataloader_prefetch_factor=config.memory.dataloader_prefetch_factor if config.memory else 2,
        report_to=config.logging.report_to,
        logging_dir=config.logging.tensorboard_log_dir,
        push_to_hub=False,  # We handle this manually via callback
        hub_model_id=config.huggingface.repo_id,
        hub_token=config.huggingface.token,
        remove_unused_columns=True,
    )

    return training_args


def find_resume_checkpoint(config: Config, logger: logging.Logger) -> Optional[str]:
    """Find the latest checkpoint to resume from."""
    output_dir = Path(config.training.output_dir)

    if not output_dir.exists():
        return None

    # Check for explicit resume path
    if config.training.resume_from_checkpoint:
        resume_path = Path(config.training.resume_from_checkpoint)
        if resume_path.exists():
            logger.info(f"Resuming from specified checkpoint: {resume_path}")
            return str(resume_path)

    # Find latest checkpoint
    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint:
        logger.info(f"Found checkpoint to resume from: {last_checkpoint}")
        return last_checkpoint

    return None


def train(config: Config, logger: logging.Logger):
    """Main training function."""
    logger.info("Starting pretraining pipeline...")

    # Set seed for reproducibility
    set_seed(config.training.seed)

    # Authenticate with Hugging Face
    if config.huggingface.token:
        logger.info("Logging into Hugging Face Hub...")
        login(token=config.huggingface.token)
    else:
        logger.warning("No HF_TOKEN found. Hub uploads will fail!")

    # Load tokenizer (pretrained)
    tokenizer = load_tokenizer(config, logger)

    # Update model config with tokenizer info
    config.model.vocab_size = tokenizer.vocab_size
    config.model.bos_token_id = tokenizer.bos_token_id or 1
    config.model.eos_token_id = tokenizer.eos_token_id or 2
    config.model.pad_token_id = tokenizer.pad_token_id or 0

    # Create model from scratch
    model = create_model_from_scratch(config, logger)

    # Enable gradient checkpointing
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Load and prepare datasets
    dataset = load_and_prepare_datasets(config, tokenizer, logger)

    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, config, logger)

    # Split into train/eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.01, seed=42)

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Get training arguments
    training_args = get_training_arguments(config)

    # Create Hub upload callback
    hub_callback = HubUploadCallback(config, logger)

    # Check for resume checkpoint
    resume_checkpoint = find_resume_checkpoint(config, logger)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[hub_callback],
    )

    # Print training info
    num_params = config.model.get_num_parameters()
    effective_batch_size = (
        config.training.per_device_train_batch_size *
        config.training.gradient_accumulation_steps *
        max(1, torch.cuda.device_count())
    )

    logger.info("\n" + "=" * 60)
    logger.info("STARTING PRETRAINING FROM SCRATCH")
    logger.info("=" * 60)
    logger.info(f"  Model: {config.model.model_name}")
    logger.info(f"  Parameters: {num_params:,} ({num_params/1e9:.2f}B)")
    logger.info(f"  Training samples: {len(split_dataset['train'])}")
    logger.info(f"  Eval samples: {len(split_dataset['test'])}")
    logger.info(f"  Max steps: {config.training.max_steps}")
    logger.info(f"  Batch size per device: {config.training.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {effective_batch_size}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Warmup steps: {config.training.warmup_steps}")
    logger.info(f"  Output directory: {config.training.output_dir}")
    logger.info(f"  Hub repository: {config.huggingface.repo_id}")
    logger.info("=" * 60 + "\n")

    # Train
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final model
    logger.info("Saving final model...")
    final_model_path = Path(config.training.output_dir) / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    # Save model config
    model.config.save_pretrained(str(final_model_path))

    logger.info("\n" + "=" * 60)
    logger.info("PRETRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

    return trainer


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pretrain a ~1B parameter LLM from scratch"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom configuration file (JSON)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override maximum training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Disable pushing to Hugging Face Hub",
    )
    parser.add_argument(
        "--memory-mode",
        type=str,
        choices=["low", "medium", "high", "auto"],
        default=None,
        help=(
            "GPU memory mode: "
            "'low' for 16GB GPU + 64GB RAM, "
            "'medium' for 46GB GPU, "
            "'high' for 80GB GPU, "
            "'auto' to detect based on available VRAM"
        ),
    )
    parser.add_argument(
        "--show-memory-modes",
        action="store_true",
        help="Show available GPU memory modes and exit",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Handle --show-memory-modes flag
    if args.show_memory_modes:
        print_memory_modes_summary()
        sys.exit(0)

    # Load configuration with optional memory mode
    if args.memory_mode == "auto":
        # Auto-detect based on available VRAM
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Detected GPU VRAM: {vram_gb:.1f} GB")
            config = get_config_for_vram(int(vram_gb))
        else:
            print("No GPU detected. Using LOW_VRAM mode with CPU offloading.")
            config = get_config(memory_mode=GPUMemoryMode.LOW_VRAM)
    elif args.memory_mode:
        # Map string to enum
        mode_map = {
            "low": GPUMemoryMode.LOW_VRAM,
            "medium": GPUMemoryMode.MEDIUM_VRAM,
            "high": GPUMemoryMode.HIGH_VRAM,
        }
        config = get_config(memory_mode=mode_map[args.memory_mode])
    else:
        # Default configuration (no memory mode applied)
        config = get_config()

    # Apply command line overrides (after memory mode to allow fine-tuning)
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.no_push:
        config.huggingface.push_to_hub = False
        config.training.push_to_hub = False

    # Setup logging
    logger = setup_logging(config)

    # Print configuration
    print_config(config)

    # Check for HF token
    if not config.huggingface.token:
        logger.error(
            "HF_TOKEN environment variable not set!\n"
            "Please set it with: export HF_TOKEN='your_token_here'"
        )
        if config.huggingface.push_to_hub:
            logger.error("Hub uploads are enabled but will fail without a token.")
            sys.exit(1)

    # Check for CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        logger.warning("CUDA not available. Training will be very slow on CPU!")

    try:
        # Run training
        trainer = train(config, logger)
        logger.info("Pretraining completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user. Checkpoint saved.")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
