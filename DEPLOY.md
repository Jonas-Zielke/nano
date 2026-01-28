# Deployment Plan for Nano ML Training

## Target System
- **Server**: Ubuntu 22.04 LTS (Proxmox VM)
- **GPU**: NVIDIA A6000 (48 GB VRAM)
- **Python**: 3.10.12

---

## Phase 1: System Dependencies

### Step 1.1: Install Required System Packages
```bash
sudo apt update
sudo apt install -y python3.10-venv python3-pip git
```

### Step 1.2: Verify NVIDIA Drivers & CUDA
```bash
# Check GPU is recognized
nvidia-smi

# Expected output should show:
# - NVIDIA RTX A6000
# - CUDA Version: 11.x or 12.x
# - ~48 GB Memory
```

If `nvidia-smi` fails, install NVIDIA drivers:
```bash
sudo apt install -y nvidia-driver-535
sudo reboot
```

---

## Phase 2: Project Setup

### Step 2.1: Clone Repository (if not already done)
```bash
cd ~
git clone https://github.com/Jonas-Zielke/nano.git
cd nano
```

### Step 2.2: Run Setup Script
```bash
chmod +x setup.sh
./setup.sh
```

This will:
1. Create a Python virtual environment
2. Install PyTorch with CUDA support (auto-detected)
3. Install all dependencies from `requirements.txt`
4. Create directory structure
5. Run import tests

### Step 2.3: Activate Virtual Environment
```bash
source venv/bin/activate
```

---

## Phase 3: Configuration

### Step 3.1: Set Hugging Face Token
```bash
# Get your token from: https://huggingface.co/settings/tokens
# Token must have WRITE access for uploading checkpoints

export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

# Or add to .bashrc for persistence:
echo 'export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
```

### Step 3.2: Verify Configuration (Optional)
```bash
# Show available GPU memory modes
python train.py --show-memory-modes

# Test with medium VRAM mode (for A6000)
python config.py medium
```

---

## Phase 4: Training

### Option A: Auto-Detect GPU (Recommended)
```bash
# Automatically selects MEDIUM_VRAM mode for your A6000
python train.py --memory-mode auto
```

### Option B: Explicit Medium VRAM Mode
```bash
# Explicitly set for 46-48 GB GPU
python train.py --memory-mode medium
```

### Option C: Custom Training Parameters
```bash
# Override specific parameters
python train.py \
    --memory-mode medium \
    --max-steps 50000 \
    --batch-size 4 \
    --learning-rate 3e-4
```

---

## Training Configuration for A6000

The `MEDIUM_VRAM` mode is pre-configured for your A6000:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 4 | Per-device training batch |
| Gradient Accumulation | 32 | Steps to accumulate |
| Effective Batch Size | 128 | 4 * 32 = 128 |
| Gradient Checkpointing | Enabled | Trades compute for memory |
| Flash Attention 2 | Enabled | Efficient attention |
| KV Heads (GQA) | 8 | Grouped Query Attention |
| Precision | bf16 | Brain floating point |
| Max Sequence Length | 2048 | Tokens per sample |

**Expected VRAM Usage**: ~38-42 GB (leaving headroom)

---

## Phase 5: Monitoring

### TensorBoard (Local)
```bash
# In a separate terminal
source venv/bin/activate
tensorboard --logdir=logs/tensorboard --port 6006

# Access at: http://localhost:6006
```

### Check Training Progress
```bash
# View latest log
tail -f logs/pretraining_*.log

# Check checkpoints
ls -la checkpoints/
```

### Monitor GPU Usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi
```

---

## Phase 6: Resume Training

If training is interrupted:
```bash
# Automatically resume from latest checkpoint
python train.py --resume --memory-mode medium
```

---

## Quick Start Commands (Copy-Paste Ready)

```bash
# Complete setup sequence for fresh server
sudo apt update && sudo apt install -y python3.10-venv python3-pip git

cd ~/nano
chmod +x setup.sh
./setup.sh

source venv/bin/activate

export HF_TOKEN='your_token_here'

python train.py --memory-mode medium
```

---

## Troubleshooting

### Issue: `python3.10-venv` not found
```bash
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10-venv
```

### Issue: CUDA not detected
```bash
# Check CUDA installation
nvcc --version

# If missing, install CUDA toolkit
sudo apt install -y nvidia-cuda-toolkit
```

### Issue: Out of Memory (OOM)
```bash
# Reduce batch size
python train.py --memory-mode medium --batch-size 2

# Or use low VRAM mode (more aggressive optimization)
python train.py --memory-mode low
```

### Issue: `bitsandbytes` errors
```bash
pip install bitsandbytes --upgrade
```

### Issue: Flash Attention not working
```bash
# Flash Attention 2 requires Ampere+ GPU (A6000 is supported)
pip install flash-attn --no-build-isolation
```

---

## Expected Training Timeline

| Phase | Steps | Duration (Estimated) |
|-------|-------|---------------------|
| Warmup | 0-2,000 | ~2-3 hours |
| Early Training | 2,000-25,000 | ~24 hours |
| Mid Training | 25,000-75,000 | ~48 hours |
| Final Training | 75,000-100,000 | ~24 hours |

**Total**: ~4-5 days for 100k steps

---

## Checkpoints & Hub Uploads

- Checkpoints saved every **5,000 steps** to `checkpoints/`
- Automatically uploaded to Hugging Face Hub: `Kanonenbombe/nano`
- Keeps last **5 checkpoints** to save disk space
- Final model saved to `checkpoints/final_model/`

---

## Safety Notes

1. **Screen/tmux**: Run training in a screen or tmux session to prevent accidental termination:
   ```bash
   tmux new -s training
   # Run training commands
   # Detach with Ctrl+B, then D
   # Reattach with: tmux attach -t training
   ```

2. **Disk Space**: Ensure at least 100 GB free for checkpoints and datasets

3. **Cooling**: A6000 will run at high utilization; ensure adequate cooling

4. **Power**: Training is interruptible; checkpoints enable safe resume
