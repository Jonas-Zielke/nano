# Nano - Train a 1B Parameter LLM from Scratch

A complete Python project for training a ~1 billion parameter language model **from scratch** (not fine-tuning) with automatic checkpointing to Hugging Face Hub.

## Features

- **Training from Scratch**: Randomly initialized weights, no pretrained model
- **LLaMA-style Architecture**: Modern transformer with RoPE, RMSNorm, SwiGLU
- **~1B Parameters**: Configurable model size (default: 1.1B parameters)
- **Multilingual Training**: German and English text corpora
- **Code Generation**: Python code from The Stack
- **Reasoning Data**: Mathematical and instruction-following datasets
- **Automatic Checkpointing**: Uploads to Hugging Face Hub during training
- **Resumable Training**: Continue from any checkpoint

## Model Architecture

The model uses a LLaMA-style architecture:

| Component | Configuration |
|-----------|--------------|
| Architecture | LLaMA (decoder-only transformer) |
| Parameters | ~1.1 billion |
| Hidden Size | 2048 |
| Layers | 22 |
| Attention Heads | 16 |
| FFN Size | 5504 (SwiGLU) |
| Context Length | 2048 tokens |
| Vocabulary | 32,000 (LLaMA tokenizer) |
| Position Encoding | RoPE |
| Normalization | RMSNorm |

## Project Structure

```
nano/
├── setup.sh           # Environment setup script
├── requirements.txt   # Python dependencies
├── config.py          # Model architecture & training config
├── train.py           # Main pretraining script
├── README.md          # This file
├── models/            # Downloaded model files
├── datasets/          # Cached datasets
├── checkpoints/       # Training checkpoints
├── scripts/           # Utility scripts
└── logs/              # Training logs
```

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/nano.git
cd nano

# Make setup script executable
chmod +x setup.sh

# Run setup (creates venv and installs dependencies)
./setup.sh
```

### 2. Activate Environment

```bash
source venv/bin/activate
```

### 3. Configure Hugging Face Token

```bash
# Set your Hugging Face token (required for dataset access and uploads)
export HF_TOKEN='your_huggingface_token_here'
```

### 4. Start Training

```bash
python train.py
```

## Configuration

All configuration is in `config.py`. Key settings:

### Model Architecture

```python
# ~1B parameter configuration
vocab_size = 32000           # Vocabulary size
hidden_size = 2048           # Hidden dimension
intermediate_size = 5504     # FFN size (SwiGLU)
num_hidden_layers = 22       # Transformer layers
num_attention_heads = 16     # Attention heads
max_position_embeddings = 2048  # Context length
```

### Training Configuration

```python
max_steps = 100000                    # Total training steps
per_device_train_batch_size = 8       # Batch size per GPU
gradient_accumulation_steps = 16      # Effective batch = 128
learning_rate = 3e-4                  # Peak learning rate
warmup_steps = 2000                   # Linear warmup steps
weight_decay = 0.1                    # AdamW weight decay
```

### Hub Configuration

```python
repo_id = "Kanonenbombe/nano"         # Destination repository
push_to_hub = True                     # Enable automatic uploads
save_steps = 5000                      # Checkpoint every N steps
```

## Command Line Options

```bash
# Basic training
python train.py

# Resume from checkpoint
python train.py --resume

# Override max steps
python train.py --max-steps 50000

# Override batch size
python train.py --batch-size 4

# Override learning rate
python train.py --learning-rate 1e-4

# Disable Hub uploads (train locally only)
python train.py --no-push
```

## Datasets

Training uses a carefully curated mix emphasizing **reasoning and chain-of-thought**:

### General Text (30%)
| Dataset | Content | Weight |
|---------|---------|--------|
| FineWeb-Edu | English educational text | 20% |
| CulturaX (German) | German web text | 10% |

### Code (10%)
| Dataset | Content | Weight |
|---------|---------|--------|
| The Stack | Python source code | 10% |

### Reasoning & Chain-of-Thought (50%)
| Dataset | Content | Weight |
|---------|---------|--------|
| GSM8K | Math word problems with step-by-step solutions | 8% |
| MetaMathQA | Mathematical reasoning with CoT | 10% |
| OpenOrca | GPT-4 reasoning explanations | 12% |
| Open-Platypus | STEM reasoning (science, math, logic) | 8% |
| Orca-Math | Math problems with detailed solutions | 7% |
| CAMEL-Math | Scientific/mathematical reasoning | 5% |

### Instruction Following (10%)
| Dataset | Content | Weight |
|---------|---------|--------|
| UltraChat | Multi-turn conversations | 5% |
| OpenSchnabeltier | German instructions | 5% |

### Reasoning Format
Data is formatted with special tokens to encourage step-by-step thinking:
```
<|user|>
What is 15% of 80?
<|assistant|>
<think>
To find 15% of 80:
Step 1: Convert 15% to decimal: 15/100 = 0.15
Step 2: Multiply: 0.15 × 80 = 12
</think>

The answer is: 12
<|end|>
```

Datasets are streamed and cached automatically on first run.

## Hardware Requirements

### Minimum (with gradient checkpointing)
- GPU: 24GB VRAM (RTX 3090, RTX 4090, A5000)
- RAM: 32GB
- Storage: 100GB for datasets and checkpoints

### Recommended
- GPU: 40GB+ VRAM (A100, H100) or multi-GPU
- RAM: 64GB
- Storage: 500GB SSD

### Multi-GPU Training

The script automatically uses all available GPUs via Hugging Face Accelerate:

```bash
# For multi-GPU training
accelerate launch train.py
```

## Training Progress

Training is logged to TensorBoard:

```bash
# View training metrics
tensorboard --logdir logs/tensorboard
```

Checkpoints are saved every 5,000 steps and uploaded to Hugging Face Hub.

## Checkpointing & Resume

### Automatic Checkpointing
- Saves model weights every `save_steps` (default: 5000)
- Uploads to Hugging Face Hub after each save
- Keeps last 5 checkpoints locally

### Resume Training

```bash
# Automatically finds latest checkpoint
python train.py --resume
```

## Expected Training Time

| Hardware | Steps/Hour | Time for 100K steps |
|----------|------------|---------------------|
| 1x RTX 4090 | ~50 | ~80 hours |
| 1x A100 40GB | ~120 | ~35 hours |
| 4x A100 40GB | ~400 | ~10 hours |
| 8x H100 | ~1000 | ~4 hours |

*Estimates vary based on batch size and sequence length.*

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size or enable gradient checkpointing:

```bash
python train.py --batch-size 4
```

Or in `config.py`:
```python
per_device_train_batch_size = 4
gradient_checkpointing = True  # Already enabled by default
```

### Hub Upload Failures

Ensure your token has write access:

```bash
huggingface-cli login
huggingface-cli whoami
```

### Slow Dataset Loading

Datasets are streamed - first run may be slow. Subsequent runs use cached data.

### CUDA Version Issues

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Architecture inspired by [LLaMA](https://arxiv.org/abs/2302.13971)
- Training setup based on [Hugging Face Transformers](https://github.com/huggingface/transformers)
- Datasets from [Hugging Face Hub](https://huggingface.co/datasets)
