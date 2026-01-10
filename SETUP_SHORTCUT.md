# H100 OpenCharacterTraining Setup Shortcut

**Use this prompt to skip troubleshooting next time:**

---

"I am setting up `OpenCharacterTraining` on an H100 node (Ubuntu/Linux, Python 3.10, CUDA 12.x). 
I want to use **PyTorch 2.4.0** and the **prebuilt Flash Attention wheel** to avoid compiling from source (which is expensive).

Please execute the following optimized setup commands immediately:

```bash
# 1. Clean existing environment (if needed)
pip uninstall -y torch torchvision torchaudio vllm flash-attn

# 2. Install PyTorch 2.4.0 compatible with CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Install VLLM version compatible with Torch 2.4.0
pip install vllm==0.6.1.post2

# 4. Install Prebuilt Flash Attention 2 (ABI False for Torch 2.4.0)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 5. Fix common dependency conflicts proactively
pip install ml_dtypes --upgrade
pip install tensorflow-cpu
pip install --upgrade scipy scikit-learn pandas matplotlib

# 6. Install the repo packages
# (Ensure you are in the repo root)
pip install -e .
pip install -e openrlhf
```
After this, check imports:
```bash
python -c "import torch; print(f'Torch: {torch.__version__}, ABI: {torch.compiled_with_cxx11_abi()}')"
python -c "import vllm; print(f'VLLM: {vllm.__version__}')"
python -c "import flash_attn; print(f'FlashAttn: {flash_attn.__version__}')"
```
---
