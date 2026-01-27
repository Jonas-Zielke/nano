"""
Configuration file for ML Training Environment - Training from Scratch
======================================================================

This module contains all hyperparameters, paths, and settings for training
a ~1B parameter language model from scratch (not fine-tuning).

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
# MODEL ARCHITECTURE CONFIGURATION (Training from Scratch)
# =============================================================================

@dataclass
class ModelConfig:
    """
    Configuration for the model architecture.

    This defines a LLaMA-style transformer architecture with approximately
    1 billion parameters, trained from scratch with randomly initialized weights.

    Architecture: LLaMA-style with:
    - RMSNorm (pre-normalization)
    - Rotary Position Embeddings (RoPE)
    - SwiGLU activation in FFN
    - Grouped Query Attention (GQA) optional

    Parameter count breakdown (~1.1B):
    - Embeddings: vocab_size * hidden_size = 32000 * 2048 = 65.5M
    - Per layer: ~50M (attention + FFN)
    - 22 layers: ~1.1B total
    """

    # Model name for saving/identification
    model_name: str = "nano-1b"

    # Vocabulary size (use standard LLaMA tokenizer vocab size)
    vocab_size: int = 32000

    # Hidden dimension
    hidden_size: int = 2048

    # FFN intermediate dimension (typically 2.7x hidden for SwiGLU)
    intermediate_size: int = 5504

    # Number of transformer layers
    num_hidden_layers: int = 22

    # Number of attention heads
    num_attention_heads: int = 16

    # Number of key-value heads (for Grouped Query Attention)
    # Set equal to num_attention_heads for standard MHA
    # Set to smaller value (e.g., 4 or 8) for GQA to save memory
    num_key_value_heads: int = 16

    # Maximum sequence length
    max_position_embeddings: int = 2048

    # RoPE base frequency
    rope_theta: float = 10000.0

    # Normalization epsilon
    rms_norm_eps: float = 1e-6

    # Dropout (set to 0 for pretraining, use for fine-tuning)
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Activation function for FFN
    hidden_act: str = "silu"  # SwiGLU uses SiLU (Swish) activation

    # Tie word embeddings (input and output)
    tie_word_embeddings: bool = True

    # Initialization standard deviation
    initializer_range: float = 0.02

    # Use bias in linear layers
    attention_bias: bool = False
    mlp_bias: bool = False

    # BOS/EOS/PAD token IDs (will be set from tokenizer)
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    # Precision for training
    torch_dtype: str = "bfloat16"

    # Use Flash Attention 2 if available
    use_flash_attention_2: bool = True

    def get_num_parameters(self) -> int:
        """Calculate approximate number of parameters."""
        # Embedding parameters
        embed_params = self.vocab_size * self.hidden_size

        # Per-layer parameters
        # Attention: Q, K, V, O projections
        attn_params = (
            self.hidden_size * self.hidden_size +  # Q
            self.hidden_size * (self.hidden_size // self.num_attention_heads * self.num_key_value_heads) +  # K
            self.hidden_size * (self.hidden_size // self.num_attention_heads * self.num_key_value_heads) +  # V
            self.hidden_size * self.hidden_size  # O
        )

        # FFN: gate, up, down projections (SwiGLU)
        ffn_params = 3 * self.hidden_size * self.intermediate_size

        # Layer norm parameters (2 per layer: attention and FFN)
        norm_params = 2 * self.hidden_size

        # Total per layer
        layer_params = attn_params + ffn_params + norm_params

        # Total model
        total = embed_params + (self.num_hidden_layers * layer_params)

        # Final layer norm
        total += self.hidden_size

        # Output projection (if not tied)
        if not self.tie_word_embeddings:
            total += self.vocab_size * self.hidden_size

        return total


# =============================================================================
# TOKENIZER CONFIGURATION
# =============================================================================

@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer."""

    # Use a pretrained tokenizer (we train model from scratch, not tokenizer)
    # LLaMA tokenizer is good for multilingual + code
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"

    # Alternative tokenizers:
    # tokenizer_name: str = "mistralai/Mistral-7B-v0.1"
    # tokenizer_name: str = "EleutherAI/gpt-neox-20b"

    # Special tokens
    add_bos_token: bool = True
    add_eos_token: bool = True

    # Padding side
    padding_side: str = "right"

    # Truncation side
    truncation_side: str = "right"


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for training datasets."""

    # Datasets for pretraining from scratch
    # For pretraining, we need large-scale text data
    datasets: List[dict] = field(default_factory=lambda: [
        {
            # Large English text corpus
            "name": "HuggingFaceFW/fineweb-edu",
            "config": "sample-10BT",
            "split": "train",
            "text_field": "text",
            "weight": 0.35,
            "streaming": True,
            "max_samples": 1000000,  # Limit for manageable training
        },
        {
            # German text corpus
            "name": "uonlp/CulturaX",
            "config": "de",
            "split": "train",
            "text_field": "text",
            "weight": 0.20,
            "streaming": True,
            "max_samples": 500000,
        },
        {
            # Code corpus (Python)
            "name": "bigcode/the-stack-dedup",
            "config": "data/python",
            "split": "train",
            "text_field": "content",
            "weight": 0.20,
            "streaming": True,
            "max_samples": 500000,
        },
        {
            # Mathematical/reasoning data
            "name": "open-web-math/open-web-math",
            "config": None,
            "split": "train",
            "text_field": "text",
            "weight": 0.15,
            "streaming": True,
            "max_samples": 300000,
        },
        {
            # Instruction/conversation data for reasoning
            "name": "HuggingFaceH4/ultrachat_200k",
            "config": None,
            "split": "train_sft",
            "text_field": "messages",
            "weight": 0.10,
            "streaming": False,
            "max_samples": 200000,
        },
    ])

    # Dataset cache directory
    cache_dir: str = str(DATASETS_DIR)

    # Maximum sequence length for training
    max_seq_length: int = 2048

    # Number of workers for data loading
    num_workers: int = 4

    # Preprocessing settings
    remove_unused_columns: bool = True

    # Shuffle buffer size for streaming datasets
    shuffle_buffer_size: int = 10000

    # Seed for reproducibility
    seed: int = 42


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    # Output directory for checkpoints
    output_dir: str = str(CHECKPOINTS_DIR)

    # Training hyperparameters for pretraining from scratch
    # Note: Pretraining typically uses more epochs/steps than fine-tuning
    num_train_epochs: int = 1  # For pretraining, often measure in steps instead
    max_steps: int = 100000  # Train for 100k steps (adjust based on compute)

    # Batch size settings
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 16  # Effective batch = 8 * 16 = 128

    # Learning rate schedule (typical for pretraining)
    learning_rate: float = 3e-4  # Higher LR for training from scratch
    min_learning_rate: float = 3e-5  # 10% of max LR
    weight_decay: float = 0.1
    warmup_steps: int = 2000  # Linear warmup
    lr_scheduler_type: str = "cosine"

    # Optimizer settings
    optim: str = "adamw_torch"  # Use standard AdamW for pretraining
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95  # Lower beta2 for pretraining stability
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Precision and memory
    fp16: bool = False
    bf16: bool = True  # Use bfloat16 for training
    gradient_checkpointing: bool = True  # Save memory at cost of speed

    # Logging and saving
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 5000  # Save every 5k steps
    save_total_limit: int = 5  # Keep last 5 checkpoints

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 5000

    # Resuming training
    resume_from_checkpoint: Optional[str] = None

    # Seed for reproducibility
    seed: int = 42

    # DataLoader settings
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True

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
    wandb_project: str = "nano-pretraining"
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
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
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

        # Ensure max_position_embeddings matches max_seq_length
        self.model.max_position_embeddings = self.dataset.max_seq_length


def get_config() -> Config:
    """Get the default configuration."""
    return Config()


def print_config(config: Config):
    """Pretty print the configuration."""
    from dataclasses import asdict

    print("=" * 60)
    print("NANO-1B PRETRAINING CONFIGURATION")
    print("=" * 60)

    # Print model size
    num_params = config.model.get_num_parameters()
    print(f"\nModel Parameters: {num_params:,} ({num_params/1e9:.2f}B)")

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
