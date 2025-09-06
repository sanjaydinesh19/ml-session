# ML Intro Session Setup (with PyTorch + uv + NVIDIA Drivers)

This guide will walk you through:

1. Installing NVIDIA Drivers (for GPU Users)
2. Handling Secure Boot / MOK Enrollment
3. Setting up a clean environment with [uv](https://github.com/astral-sh/uv)
4. Installing PyTorch (GPU or CPU)
5. Fix for Python Version

---

## 1. Install NVIDIA Driver (GPU Users only)
Upgrade your system

```
sudo apt update && sudo apt upgrade -y
```

Install Git (For sharing code)

```
sudo apt install git -y
git --version
```

Check if Python3 is installed

```
python3 --version
```

Verify your GPU

```
lspci | grep -i nvidia
```
If you see your NVIDIA card, you're good

Install NVIDIA Drivers

```
sudo apt update
sudo apt install ubuntu-drivers-common
ubuntu-drivers devices
```

This will recommend a driver. Eg: nvidia-driver-570   (distro non-free recommended)

Install the recommended driver

```
sudo apt install <driver-name> -y
```

Example:
```
sudo apt install nvidia-driver-570 -y
```

During installation, you might face a secure boot issue. If Secure Boot is enabled in UEFI, Ubuntu wont load unsigned modules 
unless you enroll a MOK (Machine Owner Key). This might halt your installation

If this happens, enter "Y" for confirmation and set a password (If prompted to do so)

Reboot your system

```
sudo reboot
```

On reboot, you should see the MOK Manager screen (blue background).

    Select Enroll MOK.

    Choose Continue.

    Confirm with Yes.

    Enter the password you created during installation (if prompted).

After this the system will boot manually, and NVIDIA Driver should be active

To verify the driver

```
nvidia-smi
```

You should see your GPU name and driver version.

## 2. Creating a Virtual Environment

uv (the new fast Python package manager / environment tool) will make your setup way cleaner and avoid pip’s dependency hell.

Install uv
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload your shell
```
exec $SHELL
```

Verify
```
uv --version
```

Inside your project folder,
```
uv venv
```

Activate it
```
source .venv/bin/activate
```

## 3. Installing PyTorch

Option A. If you have NVIDIA GPU

Verify driver & CUDA support:

```
nvidia-smi
```

Look for the CUDA Version in the top-right corner.

Match it to the closest PyTorch wheel version available:

CUDA 11.8 → use +cu118

CUDA 12.1 → use +cu121

Install PyTorch with CUDA. Example:

```
uv pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Refer this [link](https://pytorch.org/get-started/locally/) to get the exact installation command for PyTorch for your Python Version and GPU

Option B. If you do NOT have NVIDIA GPU

Install the CPU-only version:

```
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

* Note: You might run into a `networkx` version issue. If you get that then run 
```
uv pip install "networkx<=3.1"
```

## 4. Installing Packages

To install the packages needed for this session

```
uv pip install matplotlib numpy scikit-learn
```

## 5. Verify Installation

Run this:

```
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Example Output (GPU) : 

```
2.0.1+cu118 11.8 True NVIDIA GeForce RTX 3060
```

Example Output (CPU) :

```
2.0.1+cpu None False CPU only
```

## 6. Enable Jupyter Notebook Support (ipykernel)

To use this environment in Jupyter / VSCode notebooks, install ipykernel and register it:

```
uv pip install ipykernel
python -m ipykernel install --user --name=ml-session --display-name "Python (ml-session)"
```

Verify it registered:

```
jupyter kernelspec list
```

You should see:

```
ml-session   /home/<username>/.local/share/jupyter/kernels/ml-session
```

## 7. In VSCode

Open your .ipynb notebook.

Top-right → Select Kernel → choose Python (ml-session).

Run:

```
import sys, torch
print(sys.executable)
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
```
Should print the path to .venv/bin/python and your Torch info.