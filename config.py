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
from enum import Enum


# =============================================================================
# GPU MEMORY MODES
# =============================================================================

class GPUMemoryMode(Enum):
    """
    GPU memory modes for different hardware configurations.

    LOW_VRAM:    16 GB GPU + 64 GB RAM - Aggressive memory optimization
    MEDIUM_VRAM: 46 GB GPU (A6000, etc.) - Balanced performance
    HIGH_VRAM:   80 GB GPU (A100, H100) - Maximum throughput
    """
    LOW_VRAM = "low_vram"
    MEDIUM_VRAM = "medium_vram"
    HIGH_VRAM = "high_vram"


@dataclass
class MemoryConfig:
    """
    Memory-specific configuration for different GPU setups.

    This dataclass contains all settings that affect VRAM usage and
    can be tuned based on available hardware.
    """

    # Batch size settings
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 16

    # Memory optimization flags
    gradient_checkpointing: bool = True

    # CPU offloading (requires DeepSpeed or FSDP)
    cpu_offload_optimizer: bool = False
    cpu_offload_params: bool = False

    # 8-bit optimizer (requires bitsandbytes)
    use_8bit_optimizer: bool = False

    # Precision settings
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True  # Enable TF32 on Ampere+ GPUs

    # DataLoader optimization
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2

    # Model parallelism (for multi-GPU)
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"

    # Flash Attention
    use_flash_attention_2: bool = True

    # Grouped Query Attention (reduce KV cache memory)
    num_key_value_heads: int = 16  # Set lower (4, 8) to save memory

    # Sequence length (can reduce to save memory)
    max_seq_length: int = 2048

    # Description for logging
    mode_description: str = ""


# =============================================================================
# GPU MEMORY MODE PRESETS
# =============================================================================

def get_low_vram_config() -> MemoryConfig:
    """
    Configuration for 16 GB GPU + 64 GB RAM.

    Optimizations:
    - Minimal batch size (1) to fit in 16GB VRAM
    - High gradient accumulation (128) to maintain effective batch size
    - CPU offloading enabled for optimizer states
    - 8-bit AdamW optimizer to reduce memory by ~50%
    - Gradient checkpointing enabled (recompute activations)
    - Reduced KV heads (GQA) to save memory
    - Flash Attention 2 required

    Expected VRAM usage: ~14-15 GB
    Expected RAM usage: ~40-50 GB (optimizer states offloaded)
    """
    return MemoryConfig(
        # Minimal batch size - fits ~1.1B model in 16GB
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=128,  # Effective batch = 128

        # Essential memory optimizations
        gradient_checkpointing=True,

        # CPU offloading - moves optimizer states to RAM
        cpu_offload_optimizer=True,
        cpu_offload_params=False,  # Keep params on GPU for speed

        # 8-bit optimizer - halves optimizer memory
        use_8bit_optimizer=True,

        # Precision - bf16 is most memory efficient
        fp16=False,
        bf16=True,
        tf32=True,

        # Fewer workers to save system memory
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,

        # No FSDP needed for single GPU
        use_fsdp=False,
        fsdp_sharding_strategy="FULL_SHARD",

        # Flash Attention required for memory efficiency
        use_flash_attention_2=True,

        # Grouped Query Attention - 4 KV heads saves ~25% attention memory
        num_key_value_heads=4,

        # Full sequence length supported with these optimizations
        max_seq_length=2048,

        mode_description="Low VRAM Mode (16GB GPU + 64GB RAM): Aggressive memory optimization with CPU offloading"
    )


def get_medium_vram_config() -> MemoryConfig:
    """
    Configuration for 46 GB GPU (A6000, L40S, etc.).

    Optimizations:
    - Moderate batch size (4) for good GPU utilization
    - Balanced gradient accumulation (32)
    - Gradient checkpointing still enabled for safety margin
    - No CPU offloading needed
    - Standard optimizer (no 8-bit)
    - 8 KV heads for slight memory saving

    Expected VRAM usage: ~38-42 GB
    """
    return MemoryConfig(
        # Moderate batch size - good GPU utilization
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=32,  # Effective batch = 128

        # Gradient checkpointing still helpful
        gradient_checkpointing=True,

        # No CPU offloading needed
        cpu_offload_optimizer=False,
        cpu_offload_params=False,

        # Standard optimizer
        use_8bit_optimizer=False,

        # Precision
        fp16=False,
        bf16=True,
        tf32=True,

        # Standard DataLoader settings
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,

        # No FSDP for single GPU
        use_fsdp=False,
        fsdp_sharding_strategy="FULL_SHARD",

        # Flash Attention for efficiency
        use_flash_attention_2=True,

        # Slight GQA reduction - 8 KV heads
        num_key_value_heads=8,

        # Full sequence length
        max_seq_length=2048,

        mode_description="Medium VRAM Mode (46GB GPU): Balanced performance and memory usage"
    )


def get_high_vram_config() -> MemoryConfig:
    """
    Configuration for 80 GB GPU (A100, H100, etc.).

    Optimizations:
    - Large batch size (16) for maximum GPU throughput
    - Lower gradient accumulation (8)
    - Gradient checkpointing disabled for maximum speed
    - Full attention heads (no GQA reduction)
    - All performance optimizations enabled

    Expected VRAM usage: ~60-70 GB
    This leaves headroom for longer sequences or larger batches if needed.
    """
    return MemoryConfig(
        # Large batch size - maximize GPU utilization
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=8,  # Effective batch = 128

        # Disable gradient checkpointing for speed
        gradient_checkpointing=False,

        # No offloading needed
        cpu_offload_optimizer=False,
        cpu_offload_params=False,

        # Standard optimizer
        use_8bit_optimizer=False,

        # Precision
        fp16=False,
        bf16=True,
        tf32=True,

        # Maximum DataLoader performance
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,

        # No FSDP for single GPU
        use_fsdp=False,
        fsdp_sharding_strategy="FULL_SHARD",

        # Flash Attention for efficiency
        use_flash_attention_2=True,

        # Full attention heads - no GQA reduction needed
        num_key_value_heads=16,

        # Full sequence length
        max_seq_length=2048,

        mode_description="High VRAM Mode (80GB GPU): Maximum throughput, minimal memory optimization"
    )


def get_memory_config(mode: GPUMemoryMode) -> MemoryConfig:
    """Get memory configuration for the specified GPU mode."""
    configs = {
        GPUMemoryMode.LOW_VRAM: get_low_vram_config,
        GPUMemoryMode.MEDIUM_VRAM: get_medium_vram_config,
        GPUMemoryMode.HIGH_VRAM: get_high_vram_config,
    }
    return configs[mode]()


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
    # Using TinyLlama tokenizer - it's open (not gated) and Llama-compatible
    tokenizer_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Alternative open tokenizers:
    # tokenizer_name: str = "EleutherAI/gpt-neox-20b"  # 50k vocab, good for code
    # tokenizer_name: str = "Qwen/Qwen2.5-0.5B"  # Good multilingual support

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
    # Mix of general text, code, and REASONING datasets for "deep thinking"
    datasets: List[dict] = field(default_factory=lambda: [
        # =========== GENERAL TEXT ===========
        {
            # High-quality English educational text
            "name": "HuggingFaceFW/fineweb-edu",
            "config": "sample-10BT",
            "split": "train",
            "text_field": "text",
            "weight": 0.20,
            "streaming": True,
            "max_samples": 500000,
        },
        {
            # German text corpus
            "name": "uonlp/CulturaX",
            "config": "de",
            "split": "train",
            "text_field": "text",
            "weight": 0.10,
            "streaming": True,
            "max_samples": 300000,
        },

        # =========== CODE ===========
        {
            # Python code
            "name": "bigcode/the-stack-dedup",
            "config": "data/python",
            "split": "train",
            "text_field": "content",
            "weight": 0.10,
            "streaming": True,
            "max_samples": 300000,
        },

        # =========== REASONING & CHAIN-OF-THOUGHT ===========
        {
            # Math word problems with step-by-step solutions
            "name": "openai/gsm8k",
            "config": "main",
            "split": "train",
            "text_field": "question",
            "answer_field": "answer",
            "format": "reasoning",  # Special format for CoT
            "weight": 0.08,
            "streaming": False,
            "max_samples": 50000,
        },
        {
            # MetaMath - mathematical reasoning with chain-of-thought
            "name": "meta-math/MetaMathQA",
            "config": None,
            "split": "train",
            "text_field": "query",
            "answer_field": "response",
            "format": "reasoning",
            "weight": 0.10,
            "streaming": True,
            "max_samples": 300000,
        },
        {
            # OpenOrca - reasoning explanations from GPT-4
            "name": "Open-Orca/OpenOrca",
            "config": None,
            "split": "train",
            "text_field": "question",
            "answer_field": "response",
            "system_field": "system_prompt",
            "format": "reasoning",
            "weight": 0.12,
            "streaming": True,
            "max_samples": 400000,
        },
        {
            # Platypus - STEM reasoning (science, math, logic)
            "name": "garage-bAInd/Open-Platypus",
            "config": None,
            "split": "train",
            "text_field": "instruction",
            "answer_field": "output",
            "format": "reasoning",
            "weight": 0.08,
            "streaming": False,
            "max_samples": 50000,
        },
        {
            # Orca-Math - math word problems with detailed solutions
            "name": "microsoft/orca-math-word-problems-200k",
            "config": None,
            "split": "train",
            "text_field": "question",
            "answer_field": "answer",
            "format": "reasoning",
            "weight": 0.07,
            "streaming": False,
            "max_samples": 200000,
        },
        {
            # Scientific reasoning and explanations
            "name": "camel-ai/math",
            "config": None,
            "split": "train",
            "text_field": "message_1",
            "answer_field": "message_2",
            "format": "reasoning",
            "weight": 0.05,
            "streaming": True,
            "max_samples": 100000,
        },

        # =========== INSTRUCTION FOLLOWING ===========
        {
            # Multi-turn conversations with reasoning
            "name": "HuggingFaceH4/ultrachat_200k",
            "config": None,
            "split": "train_sft",
            "text_field": "messages",
            "weight": 0.05,
            "streaming": False,
            "max_samples": 100000,
        },
        {
            # German instructions and reasoning
            "name": "LeoLM/OpenSchnabeltier",
            "config": None,
            "split": "train",
            "text_field": "instruction",
            "answer_field": "output",
            "format": "reasoning",
            "weight": 0.05,
            "streaming": False,
            "max_samples": 50000,
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
    memory: Optional[MemoryConfig] = None
    gpu_memory_mode: Optional[GPUMemoryMode] = None

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

    def apply_memory_mode(self, mode: GPUMemoryMode) -> "Config":
        """
        Apply a GPU memory mode to this configuration.

        This updates training, model, and dataset configs based on the
        memory constraints of the specified GPU mode.

        Args:
            mode: The GPUMemoryMode to apply

        Returns:
            Self for method chaining
        """
        self.gpu_memory_mode = mode
        self.memory = get_memory_config(mode)

        # Apply to training config
        self.training.per_device_train_batch_size = self.memory.per_device_train_batch_size
        self.training.per_device_eval_batch_size = self.memory.per_device_eval_batch_size
        self.training.gradient_accumulation_steps = self.memory.gradient_accumulation_steps
        self.training.gradient_checkpointing = self.memory.gradient_checkpointing
        self.training.fp16 = self.memory.fp16
        self.training.bf16 = self.memory.bf16
        self.training.dataloader_num_workers = self.memory.dataloader_num_workers
        self.training.dataloader_pin_memory = self.memory.dataloader_pin_memory

        # Apply to model config
        self.model.num_key_value_heads = self.memory.num_key_value_heads
        self.model.use_flash_attention_2 = self.memory.use_flash_attention_2

        # Apply to dataset config
        self.dataset.max_seq_length = self.memory.max_seq_length
        self.dataset.num_workers = self.memory.dataloader_num_workers

        # Keep max_position_embeddings in sync
        self.model.max_position_embeddings = self.dataset.max_seq_length

        print(f"\n{'='*60}")
        print(f"GPU MEMORY MODE: {mode.value.upper()}")
        print(f"{'='*60}")
        print(f"{self.memory.mode_description}")
        print(f"\nSettings applied:")
        print(f"  - Batch size: {self.memory.per_device_train_batch_size}")
        print(f"  - Gradient accumulation: {self.memory.gradient_accumulation_steps}")
        print(f"  - Effective batch size: {self.memory.per_device_train_batch_size * self.memory.gradient_accumulation_steps}")
        print(f"  - Gradient checkpointing: {self.memory.gradient_checkpointing}")
        print(f"  - CPU offload optimizer: {self.memory.cpu_offload_optimizer}")
        print(f"  - 8-bit optimizer: {self.memory.use_8bit_optimizer}")
        print(f"  - KV heads (GQA): {self.memory.num_key_value_heads}")
        print(f"  - Flash Attention 2: {self.memory.use_flash_attention_2}")
        print(f"{'='*60}\n")

        return self


def get_config(memory_mode: Optional[GPUMemoryMode] = None) -> Config:
    """
    Get the configuration with optional memory mode.

    Args:
        memory_mode: Optional GPUMemoryMode to apply. If None, uses default settings.
                    Options: GPUMemoryMode.LOW_VRAM (16GB)
                            GPUMemoryMode.MEDIUM_VRAM (46GB)
                            GPUMemoryMode.HIGH_VRAM (80GB)

    Returns:
        Config object with appropriate settings for the hardware
    """
    config = Config()
    if memory_mode is not None:
        config.apply_memory_mode(memory_mode)
    return config


def get_config_for_vram(vram_gb: int) -> Config:
    """
    Automatically select the appropriate memory mode based on available VRAM.

    Args:
        vram_gb: Available GPU VRAM in gigabytes

    Returns:
        Config object with appropriate memory mode applied
    """
    if vram_gb <= 24:
        mode = GPUMemoryMode.LOW_VRAM
    elif vram_gb <= 48:
        mode = GPUMemoryMode.MEDIUM_VRAM
    else:
        mode = GPUMemoryMode.HIGH_VRAM

    return get_config(memory_mode=mode)


def print_config(config: Config):
    """Pretty print the configuration."""
    from dataclasses import asdict

    print("=" * 60)
    print("NANO-1B PRETRAINING CONFIGURATION")
    print("=" * 60)

    # Print GPU memory mode if set
    if config.gpu_memory_mode:
        print(f"\nGPU Memory Mode: {config.gpu_memory_mode.value.upper()}")
        if config.memory:
            print(f"Description: {config.memory.mode_description}")

    # Print model size
    num_params = config.model.get_num_parameters()
    print(f"\nModel Parameters: {num_params:,} ({num_params/1e9:.2f}B)")

    config_dict = asdict(config)
    for section, values in config_dict.items():
        if values is None:
            continue
        print(f"\n[{section.upper()}]")
        if isinstance(values, dict):
            for key, value in values.items():
                # Truncate long values
                str_value = str(value)
                if len(str_value) > 60:
                    str_value = str_value[:57] + "..."
                print(f"  {key}: {str_value}")

    print("\n" + "=" * 60)


def print_memory_modes_summary():
    """Print a summary of all available GPU memory modes."""
    print("=" * 70)
    print("AVAILABLE GPU MEMORY MODES")
    print("=" * 70)

    modes = [
        (GPUMemoryMode.LOW_VRAM, "16 GB GPU + 64 GB RAM"),
        (GPUMemoryMode.MEDIUM_VRAM, "46 GB GPU (A6000, L40S)"),
        (GPUMemoryMode.HIGH_VRAM, "80 GB GPU (A100, H100)"),
    ]

    for mode, hw_desc in modes:
        mem_config = get_memory_config(mode)
        effective_batch = mem_config.per_device_train_batch_size * mem_config.gradient_accumulation_steps

        print(f"\n{mode.value.upper()} - {hw_desc}")
        print("-" * 50)
        print(f"  Batch size:              {mem_config.per_device_train_batch_size}")
        print(f"  Gradient accumulation:   {mem_config.gradient_accumulation_steps}")
        print(f"  Effective batch size:    {effective_batch}")
        print(f"  Gradient checkpointing:  {mem_config.gradient_checkpointing}")
        print(f"  CPU offload optimizer:   {mem_config.cpu_offload_optimizer}")
        print(f"  8-bit optimizer:         {mem_config.use_8bit_optimizer}")
        print(f"  KV heads (GQA):          {mem_config.num_key_value_heads}")
        print(f"  Flash Attention 2:       {mem_config.use_flash_attention_2}")
        print(f"  Max sequence length:     {mem_config.max_seq_length}")

    print("\n" + "=" * 70)
    print("\nUsage examples:")
    print("  # Get config for specific mode")
    print("  config = get_config(memory_mode=GPUMemoryMode.LOW_VRAM)")
    print("")
    print("  # Auto-detect based on VRAM")
    print("  config = get_config_for_vram(16)  # For 16GB GPU")
    print("")
    print("  # Apply mode to existing config")
    print("  config = get_config()")
    print("  config.apply_memory_mode(GPUMemoryMode.HIGH_VRAM)")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    # Check for command line arguments
    if len(sys.argv) > 1:
        mode_arg = sys.argv[1].lower()
        mode_map = {
            "low": GPUMemoryMode.LOW_VRAM,
            "low_vram": GPUMemoryMode.LOW_VRAM,
            "16gb": GPUMemoryMode.LOW_VRAM,
            "medium": GPUMemoryMode.MEDIUM_VRAM,
            "medium_vram": GPUMemoryMode.MEDIUM_VRAM,
            "46gb": GPUMemoryMode.MEDIUM_VRAM,
            "high": GPUMemoryMode.HIGH_VRAM,
            "high_vram": GPUMemoryMode.HIGH_VRAM,
            "80gb": GPUMemoryMode.HIGH_VRAM,
            "summary": None,  # Special case for summary
        }

        if mode_arg == "summary":
            print_memory_modes_summary()
        elif mode_arg in mode_map:
            config = get_config(memory_mode=mode_map[mode_arg])
            print_config(config)
        else:
            print(f"Unknown mode: {mode_arg}")
            print("Available modes: low, medium, high, summary")
            print("Or VRAM sizes: 16gb, 46gb, 80gb")
            sys.exit(1)
    else:
        # Default: show summary and default config
        print_memory_modes_summary()
        print("\n\nDEFAULT CONFIGURATION (no memory mode applied):")
        config = get_config()
        print_config(config)
