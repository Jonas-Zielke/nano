# Nano - ML Training Environment

A complete Python project for training a small (~1B parameter) multilingual language model with automatic checkpointing to Hugging Face Hub.

## Features

- **Multilingual Training**: Supports German and English language learning
- **Code Generation**: Trained on code datasets for programming tasks
- **Reasoning Capabilities**: Includes chain-of-thought and mathematical reasoning datasets
- **Efficient Training**: Uses LoRA and 4-bit quantization for memory efficiency
- **Automatic Checkpointing**: Uploads model checkpoints to Hugging Face after each epoch
- **Resumable Training**: Can continue from the last checkpoint if interrupted

## Project Structure

```
nano/
├── setup.sh           # Environment setup script
├── requirements.txt   # Python dependencies
├── config.py          # Configuration settings
├── train.py           # Main training script
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
# Set your Hugging Face token
export HF_TOKEN='your_huggingface_token_here'
```

### 4. Start Training

```bash
python train.py
```

## Configuration

All configuration options are in `config.py`. Key settings include:

### Model Configuration

```python
model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Base model
use_lora = True          # Enable LoRA for efficient fine-tuning
load_in_4bit = True      # 4-bit quantization for memory efficiency
```

### Training Configuration

```python
num_train_epochs = 3                    # Number of epochs
per_device_train_batch_size = 4         # Batch size per GPU
gradient_accumulation_steps = 8         # Effective batch = 4 * 8 = 32
learning_rate = 2e-5                    # Learning rate
```

### Hub Configuration

```python
repo_id = "Kanonenbombe/nano"           # Destination repository
push_to_hub = True                       # Enable automatic uploads
```

## Command Line Options

```bash
# Basic training
python train.py

# Resume from checkpoint
python train.py --resume

# Override epochs
python train.py --epochs 5

# Override batch size
python train.py --batch-size 8

# Override learning rate
python train.py --learning-rate 1e-5

# Disable Hub uploads
python train.py --no-push
```

## Datasets

The training uses a combination of datasets:

| Dataset | Purpose | Weight |
|---------|---------|--------|
| HuggingFaceH4/ultrachat_200k | Multilingual conversations | 40% |
| LeoLM/OpenSchnabeltier | German instructions | 20% |
| bigcode/starcoderdata | Code generation | 20% |
| gsm8k | Mathematical reasoning | 20% |

Datasets are automatically downloaded and cached on first run.

## Model Architecture

- **Base Model**: TinyLlama-1.1B-Chat (1.1B parameters)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit NF4 quantization with double quantization
- **Attention**: Flash Attention 2 (if available)

## Hardware Requirements

### Minimum
- GPU: 8GB VRAM (RTX 3070, RTX 4070, etc.)
- RAM: 16GB
- Storage: 50GB for datasets and checkpoints

### Recommended
- GPU: 16GB+ VRAM (RTX 4090, A100, etc.)
- RAM: 32GB
- Storage: 100GB SSD

## Checkpointing

Checkpoints are automatically saved and uploaded after each epoch:

1. **Local Checkpoint**: Saved to `checkpoints/checkpoint-epoch-N/`
2. **Hub Upload**: Pushed to `Kanonenbombe/nano`
3. **Included Files**:
   - Model weights (LoRA adapters)
   - Tokenizer configuration
   - Training state (epoch, loss, etc.)
   - Dataset information

### Resume Training

If training is interrupted:

```bash
# Automatically finds and resumes from latest checkpoint
python train.py --resume
```

## Logging

Training logs are saved to `logs/` and include:

- Training metrics (loss, learning rate, etc.)
- TensorBoard logs for visualization
- Upload history for Hub checkpoints

### View TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size or increase gradient accumulation:

```bash
python train.py --batch-size 2
```

Or edit `config.py`:
```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 16
```

### Hub Upload Failures

Ensure your token has write access:

```bash
# Test authentication
huggingface-cli whoami
```

### CUDA Not Available

Install PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [TinyLlama](https://github.com/jzhang38/TinyLlama) for the base model
- [Hugging Face](https://huggingface.co) for transformers and datasets
- [PEFT](https://github.com/huggingface/peft) for efficient fine-tuning
