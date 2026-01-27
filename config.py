"""
Configuration file for ML Training Environment
==============================================

This module contains all hyperparameters, paths, and settings for training.
Modify these values to customize your training run.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base project directory (auto-detected)
PROJECT_ROOT = Path(__file__).parent.absolute()

# Directory structure
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for dir_path in [MODELS_DIR, DATASETS_DIR, CHECKPOINTS_DIR, SCRIPTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HUGGING FACE CONFIGURATION
# =============================================================================

@dataclass
class HuggingFaceConfig:
    """Configuration for Hugging Face Hub integration."""

    # Authentication token (set via environment variable for security)
    token: str = os.environ.get("HF_TOKEN", "")

    # Repository to upload checkpoints
    repo_id: str = "Kanonenbombe/nano"

    # Whether to push to hub after each epoch
    push_to_hub: bool = True

    # Commit message template
    commit_message_template: str = "Checkpoint after epoch {epoch} - Loss: {loss:.4f}"

    # Private repository
    private: bool = False


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for the model architecture."""

    # Base model from Hugging Face (approximately 1B parameters)
    # TinyLlama is 1.1B parameters, good for multilingual and code
    model_name_or_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Alternative models (uncomment to use):
    # model_name_or_path: str = "microsoft/phi-1_5"  # 1.3B, strong at code
    # model_name_or_path: str = "EleutherAI/pythia-1b-deduped"  # 1B general
    # model_name_or_path: str = "Qwen/Qwen1.5-0.5B"  # Smaller but multilingual

    # Model configuration
    torch_dtype: str = "bfloat16"  # Use bfloat16 for efficiency
    use_flash_attention_2: bool = True  # Enable if supported
    trust_remote_code: bool = True

    # LoRA configuration for efficient fine-tuning
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Quantization (4-bit for memory efficiency)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for training datasets."""

    # Datasets for multilingual reasoning and code generation
    # Each tuple: (dataset_name, config/subset, split)
    datasets: List[dict] = field(default_factory=lambda: [
        {
            # Multilingual instruction following and reasoning
            "name": "HuggingFaceH4/ultrachat_200k",
            "config": None,
            "split": "train_sft",
            "text_field": "messages",
            "weight": 0.4,  # 40% of training data
        },
        {
            # German reasoning and instruction dataset
            "name": "LeoLM/OpenSchnabeltier",
            "config": None,
            "split": "train",
            "text_field": "instruction",
            "weight": 0.2,  # 20% of training data
        },
        {
            # Code generation and understanding
            "name": "bigcode/starcoderdata",
            "config": "python",
            "split": "train",
            "text_field": "content",
            "weight": 0.2,  # 20% of training data
            "streaming": True,  # Large dataset, use streaming
        },
        {
            # Mathematical reasoning (chain-of-thought)
            "name": "gsm8k",
            "config": "main",
            "split": "train",
            "text_field": "question",
            "answer_field": "answer",
            "weight": 0.2,  # 20% of training data
        },
    ])

    # Dataset cache directory
    cache_dir: str = str(DATASETS_DIR)

    # Maximum sequence length
    max_seq_length: int = 2048

    # Number of workers for data loading
    num_workers: int = 4

    # Preprocessing settings
    remove_unused_columns: bool = True

    # Streaming mode for large datasets
    streaming: bool = False


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    # Output directory for checkpoints
    output_dir: str = str(CHECKPOINTS_DIR)

    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Effective batch size = 4 * 8 = 32

    # Learning rate schedule
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # Optimization
    optim: str = "paged_adamw_32bit"
    max_grad_norm: float = 1.0

    # Precision and memory
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Logging and saving
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: int = 3

    # Evaluation
    eval_strategy: str = "epoch"
    eval_steps: int = 500

    # Resuming training
    resume_from_checkpoint: Optional[str] = None

    # Seed for reproducibility
    seed: int = 42

    # DataLoader settings
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Hub integration
    push_to_hub: bool = True
    hub_model_id: str = "Kanonenbombe/nano"
    hub_strategy: str = "every_save"
    hub_private_repo: bool = False


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""

    # Logging directory
    log_dir: str = str(LOGS_DIR)

    # Logging level
    log_level: str = "INFO"

    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_log_dir: str = str(LOGS_DIR / "tensorboard")

    # Weights & Biases (optional)
    use_wandb: bool = False
    wandb_project: str = "nano-training"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Report to (can include multiple: "tensorboard", "wandb", "none")
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])


# =============================================================================
# COMBINED CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Master configuration combining all settings."""

    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure HF token is set
        if not self.huggingface.token:
            print("Warning: HF_TOKEN not set. Hub uploads will fail.")

        # Sync hub settings
        self.training.hub_model_id = self.huggingface.repo_id
        self.training.push_to_hub = self.huggingface.push_to_hub


def get_config() -> Config:
    """Get the default configuration."""
    return Config()


def print_config(config: Config):
    """Pretty print the configuration."""
    import json
    from dataclasses import asdict

    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)

    config_dict = asdict(config)
    for section, values in config_dict.items():
        print(f"\n[{section.upper()}]")
        for key, value in values.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 60:
                str_value = str_value[:57] + "..."
            print(f"  {key}: {str_value}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Print configuration when run directly
    config = get_config()
    print_config(config)
