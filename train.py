#!/usr/bin/env python3
"""
Nano ML Training Script
=======================

This script trains a small (~1B parameter) language model on multilingual
(German/English) and code generation datasets with automatic checkpointing
to Hugging Face Hub after each epoch.

Features:
- Automatic dataset downloading and caching
- Efficient training with LoRA and 4-bit quantization
- Epoch-based checkpointing with Hub uploads
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
from typing import Optional, Dict, Any, List
import time

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from huggingface_hub import HfApi, login, create_repo, upload_folder
from tqdm import tqdm

from config import get_config, print_config, Config

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(config: Config) -> logging.Logger:
    """Configure logging for the training run."""
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.logging.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger("nano-trainer")
    logger.info(f"Logging to {log_file}")

    return logger


# =============================================================================
# HUGGING FACE HUB CALLBACK
# =============================================================================

class HubUploadCallback(TrainerCallback):
    """
    Custom callback to upload checkpoints to Hugging Face Hub after each epoch.

    This callback handles:
    - Creating the repository if it doesn't exist
    - Uploading model checkpoints after each epoch
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
                    ignore_patterns=["*.bin", "optimizer.pt", "scheduler.pt"],
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

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of each epoch - upload checkpoint to Hub."""
        if not self.config.huggingface.push_to_hub:
            return

        epoch = int(state.epoch)
        current_loss = state.log_history[-1].get("loss", 0.0) if state.log_history else 0.0

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"EPOCH {epoch} COMPLETED - Starting Hub upload...")
        self.logger.info(f"{'='*60}")

        # Create checkpoint directory for this epoch
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-epoch-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save training state info
        state_info = {
            "epoch": epoch,
            "global_step": state.global_step,
            "loss": current_loss,
            "timestamp": datetime.now().isoformat(),
            "log_history": state.log_history,
        }

        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state_info, f, indent=2)

        # Save dataset info
        dataset_info = {
            "datasets_used": [d["name"] for d in self.config.dataset.datasets],
            "max_seq_length": self.config.dataset.max_seq_length,
        }
        with open(checkpoint_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

        # Format commit message
        commit_message = self.config.huggingface.commit_message_template.format(
            epoch=epoch,
            loss=current_loss,
        )

        # Upload to Hub
        success = self._upload_with_retry(
            folder_path=str(checkpoint_dir),
            commit_message=commit_message,
        )

        # Track upload history
        self.upload_history.append({
            "epoch": epoch,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "loss": current_loss,
        })

        if success:
            self.logger.info(f"Successfully uploaded checkpoint for epoch {epoch}")
        else:
            self.logger.error(f"Failed to upload checkpoint for epoch {epoch}")

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

        # Upload final checkpoint
        success = self._upload_with_retry(
            folder_path=args.output_dir,
            commit_message=f"Final model - Training completed at {datetime.now().isoformat()}",
        )

        if success:
            self.logger.info("Final model uploaded successfully!")
        else:
            self.logger.error("Failed to upload final model.")


# =============================================================================
# DATASET LOADING AND PREPROCESSING
# =============================================================================

def format_chat_template(
    example: Dict[str, Any],
    tokenizer,
    text_field: str = "messages",
) -> Dict[str, str]:
    """Format chat messages using the model's chat template."""
    if text_field == "messages" and "messages" in example:
        # Handle chat format
        messages = example["messages"]
        if isinstance(messages, list) and len(messages) > 0:
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return {"text": text}
            except Exception:
                # Fallback: concatenate messages
                text = "\n".join(
                    [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
                )
                return {"text": text}
    elif text_field in example:
        return {"text": str(example[text_field])}
    return {"text": ""}


def format_instruction_response(
    example: Dict[str, Any],
    tokenizer,
    instruction_field: str = "instruction",
    response_field: str = "response",
) -> Dict[str, str]:
    """Format instruction-response pairs."""
    instruction = example.get(instruction_field, "")
    response = example.get(response_field, example.get("output", ""))

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

    return {"text": text}


def format_code_example(
    example: Dict[str, Any],
    text_field: str = "content",
) -> Dict[str, str]:
    """Format code examples."""
    code = example.get(text_field, "")
    if isinstance(code, str):
        return {"text": f"```python\n{code}\n```"}
    return {"text": ""}


def format_math_example(
    example: Dict[str, Any],
    question_field: str = "question",
    answer_field: str = "answer",
) -> Dict[str, str]:
    """Format math reasoning examples with chain-of-thought."""
    question = example.get(question_field, "")
    answer = example.get(answer_field, "")

    text = f"### Problem:\n{question}\n\n### Solution:\n{answer}"
    return {"text": text}


def load_and_prepare_datasets(
    config: Config,
    tokenizer,
    logger: logging.Logger,
) -> Dataset:
    """Load, preprocess, and combine all datasets."""
    logger.info("Loading datasets...")

    all_datasets = []
    dataset_configs = config.dataset.datasets

    for ds_config in dataset_configs:
        ds_name = ds_config["name"]
        ds_subset = ds_config.get("config")
        ds_split = ds_config.get("split", "train")
        text_field = ds_config.get("text_field", "text")
        weight = ds_config.get("weight", 1.0)
        streaming = ds_config.get("streaming", False)

        logger.info(f"Loading dataset: {ds_name} (config: {ds_subset}, split: {ds_split})")

        try:
            # Load dataset
            if streaming:
                dataset = load_dataset(
                    ds_name,
                    ds_subset,
                    split=ds_split,
                    streaming=True,
                    cache_dir=config.dataset.cache_dir,
                    trust_remote_code=True,
                )
                # Take a subset for streaming datasets
                dataset = dataset.take(10000)
                dataset = Dataset.from_generator(lambda: dataset)
            else:
                dataset = load_dataset(
                    ds_name,
                    ds_subset,
                    split=ds_split,
                    cache_dir=config.dataset.cache_dir,
                    trust_remote_code=True,
                )

            # Apply appropriate formatting based on dataset type
            if "ultrachat" in ds_name.lower() or text_field == "messages":
                dataset = dataset.map(
                    lambda x: format_chat_template(x, tokenizer, text_field),
                    remove_columns=dataset.column_names,
                    desc=f"Formatting {ds_name}",
                )
            elif "schnabeltier" in ds_name.lower() or "instruction" in text_field:
                dataset = dataset.map(
                    lambda x: format_instruction_response(x, tokenizer),
                    remove_columns=dataset.column_names,
                    desc=f"Formatting {ds_name}",
                )
            elif "starcoder" in ds_name.lower() or "code" in ds_name.lower():
                dataset = dataset.map(
                    lambda x: format_code_example(x, text_field),
                    remove_columns=dataset.column_names,
                    desc=f"Formatting {ds_name}",
                )
            elif "gsm" in ds_name.lower() or "math" in ds_name.lower():
                dataset = dataset.map(
                    lambda x: format_math_example(
                        x,
                        ds_config.get("text_field", "question"),
                        ds_config.get("answer_field", "answer"),
                    ),
                    remove_columns=dataset.column_names,
                    desc=f"Formatting {ds_name}",
                )
            else:
                # Generic text field extraction
                dataset = dataset.map(
                    lambda x: {"text": str(x.get(text_field, ""))},
                    remove_columns=dataset.column_names,
                    desc=f"Formatting {ds_name}",
                )

            # Filter empty examples
            dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

            # Sample based on weight
            if weight < 1.0:
                num_samples = int(len(dataset) * weight)
                dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

            logger.info(f"  Loaded {len(dataset)} examples from {ds_name}")
            all_datasets.append(dataset)

        except Exception as e:
            logger.warning(f"Failed to load dataset {ds_name}: {e}")
            continue

    if not all_datasets:
        raise ValueError("No datasets were loaded successfully!")

    # Combine all datasets
    logger.info("Combining datasets...")
    combined_dataset = concatenate_datasets(all_datasets)
    combined_dataset = combined_dataset.shuffle(seed=42)

    logger.info(f"Total combined dataset size: {len(combined_dataset)} examples")

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

    logger.info(f"Tokenized dataset size: {len(tokenized_dataset)}")

    return tokenized_dataset


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_and_tokenizer(
    config: Config,
    logger: logging.Logger,
) -> tuple:
    """Load and configure the model and tokenizer."""
    logger.info(f"Loading model: {config.model.model_name_or_path}")

    # Configure quantization
    if config.model.load_in_4bit:
        logger.info("Configuring 4-bit quantization...")
        compute_dtype = getattr(torch, config.model.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.model.bnb_4bit_use_double_quant,
        )
    else:
        bnb_config = None

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name_or_path,
        trust_remote_code=config.model.trust_remote_code,
        token=config.huggingface.token,
    )

    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    logger.info("Loading model...")
    model_kwargs = {
        "trust_remote_code": config.model.trust_remote_code,
        "token": config.huggingface.token,
        "device_map": "auto",
    }

    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    # Try to use flash attention if available
    if config.model.use_flash_attention_2:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        except Exception:
            logger.warning("Flash Attention 2 not available, using default attention")

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name_or_path,
        **model_kwargs,
    )

    # Prepare model for k-bit training if using quantization
    if config.model.load_in_4bit:
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
        )

    # Apply LoRA if configured
    if config.model.use_lora:
        logger.info("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=config.model.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Enable gradient checkpointing
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    logger.info("Model and tokenizer loaded successfully!")

    return model, tokenizer


# =============================================================================
# TRAINING
# =============================================================================

def get_training_arguments(config: Config) -> TrainingArguments:
    """Create training arguments from configuration."""
    return TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        optim=config.training.optim,
        max_grad_norm=config.training.max_grad_norm,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
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
        report_to=config.logging.report_to,
        logging_dir=config.logging.tensorboard_log_dir,
        push_to_hub=False,  # We handle this manually via callback
        hub_model_id=config.huggingface.repo_id,
        hub_token=config.huggingface.token,
    )


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
    logger.info("Starting training pipeline...")

    # Authenticate with Hugging Face
    if config.huggingface.token:
        logger.info("Logging into Hugging Face Hub...")
        login(token=config.huggingface.token)
    else:
        logger.warning("No HF_TOKEN found. Hub uploads will fail!")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, logger)

    # Load and prepare datasets
    dataset = load_and_prepare_datasets(config, tokenizer, logger)

    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, config, logger)

    # Split into train/eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.05, seed=42)

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
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

    # Start training
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    logger.info(f"  Model: {config.model.model_name_or_path}")
    logger.info(f"  Training samples: {len(split_dataset['train'])}")
    logger.info(f"  Eval samples: {len(split_dataset['test'])}")
    logger.info(f"  Epochs: {config.training.num_train_epochs}")
    logger.info(f"  Batch size: {config.training.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
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

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

    return trainer


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a multilingual LLM with code generation capabilities"
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
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
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

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = get_config()

    # Apply command line overrides
    if args.epochs:
        config.training.num_train_epochs = args.epochs
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
    else:
        logger.warning("CUDA not available. Training will be slow on CPU.")

    try:
        # Run training
        trainer = train(config, logger)
        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user. Checkpoint saved.")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
