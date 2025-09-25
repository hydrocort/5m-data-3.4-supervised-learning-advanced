# LightGBM CUDA Installation Guide for WSL2

This guide shows you how to build and install LightGBM with CUDA support on WSL2 (Windows Subsystem for Linux) to accelerate machine learning training with your NVIDIA GPU.

## 🎯 What You'll Achieve

- **GPU-accelerated LightGBM training** using your NVIDIA GPU
- **Significantly faster training times** (5-10x speedup potential)
- **No more infinite training loops** with proper GPU acceleration
- **Full compatibility with your Python environment**

## 📋 Prerequisites

### System Requirements

- **Windows 11** or **Windows 10 version 21H2+**
- **WSL2** (not WSL1)
- **NVIDIA GPU** (GTX 10 series or newer recommended)
- **NVIDIA drivers** version 470.76 or later

### Check Your Setup

Run these commands to verify your system:

```bash
# Check WSL version
wsl --version

# Check NVIDIA driver (run from Windows PowerShell)
nvidia-smi

# Check if you're in WSL2
uname -a  # Should show "microsoft-standard-WSL2"
```

## 🔧 Step 1: Install NVIDIA Drivers (Windows Side)

1. **Download latest NVIDIA drivers** from [NVIDIA Downloads](https://www.nvidia.com/Download/index.aspx)
2. **Install the drivers** on Windows (NOT in WSL)
3. **Reboot** your computer
4. **Verify installation** by running `nvidia-smi` in Windows PowerShell

> ⚠️ **Important**: Only install drivers on Windows, never in WSL!

## 🐧 Step 2: Set Up WSL2 Environment

### Install WSL2 (if not already installed)

```bash
# Run in Windows PowerShell as Administrator
wsl --install
wsl --set-default-version 2
```

### Update WSL2 and Install Dependencies

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install build essentials
sudo apt install -y build-essential cmake git

# Install CUDA toolkit in WSL
sudo apt install -y nvidia-cuda-toolkit

# Verify CUDA installation
nvcc --version
nvidia-smi  # Should work from WSL too
```

## 🔨 Step 3: Install Compatible GCC Version

LightGBM CUDA compilation requires GCC 12 (not the default GCC 13 in Ubuntu 24.04).

```bash
# Install GCC 12
sudo apt install -y gcc-12 g++-12

# Verify installation
gcc-12 --version
g++-12 --version
```

## 🚀 Step 4: Build LightGBM with CUDA Support

### Create Build Environment

```bash
# Create temporary build directory
mkdir -p ~/temp_lightgbm_build
cd ~/temp_lightgbm_build

# Clone LightGBM source code
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
```

### Set Up Build Environment

```bash
# Set GCC 12 as compiler for this session
export CC=gcc-12
export CXX=g++-12

# Verify correct compiler versions
echo "Using GCC: $(gcc-12 --version | head -1)"
echo "Using G++: $(g++-12 --version | head -1)"
```

### Build and Install

```bash
# Build LightGBM with CUDA support
# Replace 'your-env-name' with your actual conda environment name
conda run -n your-env-name sh build-python.sh install --cuda

# This process takes 10-30 minutes depending on your system
```

## ✅ Step 5: Verify Installation

### Test CUDA Support

```bash
# Test in your conda environment
conda run -n your-env-name python -c "
import lightgbm as lgb
print(f'LightGBM version: {lgb.__version__}')

# Test CUDA functionality
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2)
model = lgb.LGBMClassifier(device='cuda', n_estimators=5, verbose=-1)
model.fit(X, y)
print('✅ CUDA LightGBM working successfully!')
"
```

## 🐍 Step 6: Update Your Python Code

### Basic Usage

```python
import lightgbm as lgb

# Use CUDA acceleration
model = lgb.LGBMClassifier(
    random_state=0,
    device='cuda',          # Enable CUDA acceleration
    verbose=-1              # Reduce output verbosity
)

# Train normally
model.fit(X_train, y_train)
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue: "CUDA Tree Learner was not enabled"

**Solution**: The pre-built pip package doesn't have CUDA support. You must build from source as shown above.

#### Issue: "unsupported GNU version! gcc versions later than 12 are not supported"

**Solution**: Install and use GCC 12 as shown in Step 3.

```bash
sudo apt install gcc-12 g++-12
export CC=gcc-12
export CXX=g++-12
```

## 🧹 Clean Up

After successful installation, you can remove the build directory:

```bash
rm -rf ~/temp_lightgbm_build
```

## 📊 Expected Performance

With a modern NVIDIA GPU (RTX 3060 or better), expect:

- **5-10x speedup** over CPU training
- **Ability to handle larger datasets** efficiently
- **Stable training** without infinite loops
- **Same model accuracy** as CPU training

## 🤝 Contributing

If you encounter issues or have improvements:

1. **Document the error** with full error messages
2. **Check your GCC version** (`gcc --version`)
3. **Verify CUDA installation** (`nvcc --version`)
4. **Share solutions** with classmates!

## 📚 Additional Resources

- [LightGBM Official Documentation](https://lightgbm.readthedocs.io/)
- [NVIDIA CUDA on WSL Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [Microsoft WSL Documentation](https://docs.microsoft.com/en-us/windows/wsl/)

---

**Created by**: [Your Class] Data Science Team  
**Last Updated**: September 2025  
**Tested On**: Ubuntu 24.04 LTS (WSL2), CUDA 12.0, LightGBM 4.6.0+

> 💡 **Tip**: Save this guide and share it with classmates who want GPU acceleration for their machine learning projects!
