# Enabling CUDA Support for `llama.cpp` with `llama-cpp-python` and Conda

This document explains the step-by-step process of compiling and configuring `llama.cpp` to use **CUDA** on a system with an NVIDIA GPU. It also outlines the rationale behind each step and the solution to common pitfalls.

The 'issue' is that when conda installs `llama-cpp-python` it looks for libllama.so within the environment and ignores the the system build. 
To correct this we make sure there is an environmental that points to the system build and build `llama-cpp-python` with the following arguments:
```bash
CMAKE_ARGS="-DLLAMA_CMAKE_CUDA=on -DLLAMA_CPP_USE_VENDORED=OFF" \
FORCE_CMAKE=1 \
pip install git+https://github.com/abetlen/llama-cpp-python.git
```

Furthermore, there is a conflict with the `libcublas` version that is required by pytorch-cuda=12.1 and the version that is 
available in the conda-forge channel. This is resolved by installing the pytorch-cuda=12.1 package first and then installing the other packages.
Which means we can't create the environment in one go but have to do it in stages.

---

## 🔧 System Setup

- **OS**: Pop! OS 22.04 LTS  
- **CPU**: Intel Core i7-9750H @ 2.60GHz (12 threads)  
- **GPU**: NVIDIA RTX 2070 Mobile (Max-Q Design)  
- **RAM**: 32 GB

### ✅ NVIDIA and CUDA

- Installed from: https://developer.nvidia.com/cuda-downloads
- `nvcc --version`:
  ```
  Cuda compilation tools, release 12.8, V12.8.93
  ```
- `nvidia-smi`:
  ```
  NVIDIA-SMI 565.77
  Driver Version: 565.77
  CUDA Version: 12.7
  ```

---

## 🎯 Goal

- Use `llama.cpp` and `llama-cpp-python` **with GPU acceleration** using the system CUDA install.
- Avoid rebuilding `llama.cpp` for every project or Python environment.
- Integrate into a Conda-based Python ecosystem.

---

## ❌ The Problem

Installing `llama-cpp-python` via pip defaults to a precompiled binary **without CUDA support**. Even forcing builds from source will:

1. **Vendor a local copy** of `llama.cpp`, ignoring your system build.
2. Not link to your existing `llama.cpp` compiled with CUDA.

Additionally, creating a Conda environment via a unified `environment.yml` failed due to a `libcublas` version mismatch:

```
LibMambaUnsatisfiableError: Encountered problems while solving:
  - package pytorch-cuda-12.1 requires libcublas >=12.1.0.26,<12.1.3.1
```

This required us to install dependencies in **stages**, avoiding pre-resolution of incompatible versions.

---

## ✅ The Solution

### 1. Clone and Build `llama.cpp`

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_F16=ON -DGGML_CUDA_FORCE_MMQ=ON
cmake --build build --config Release
```

This will build the CUDA-enabled binaries and libraries at:
```bash
$HOME/llama.cpp/build/bin/
```

### 2. Global Conda Hook (Environment Variables)

Create a **global** activation hook so that every Conda environment will automatically pick up the correct `libllama.so`:

```bash
mkdir -p ~/anaconda3/etc/conda/activate.d

cat <<EOF > ~/anaconda3/etc/conda/activate.d/env_llama_cpp.sh
export LLAMA_CPP_LIB="$HOME/llama.cpp/build/bin/libllama.so"
EOF
```

This avoids needing to set the environment variable per project or per shell session.

### 3. Optional `.bashrc` Setup

To use `llama-run` CLI globally from any terminal:

```bash
mkdir -p ~/bin
ln -s ~/llama.cpp/build/bin/llama-run ~/bin/llama
```
Then, ensure `~/bin` is on your `PATH`:

```bash
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## 🐍 Conda Environment Setup (Staged)

We could not create the environment directly from `environment.yml` due to dependency conflicts. Instead, we used a staged approach:

### Step-by-step Manual Setup

```bash
conda create -n llama-env python=3.10
conda activate llama-env

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Scientific/ML packages
conda install -c conda-forge scikit-learn pandas numpy matplotlib scipy geopandas polars streamlit

# Python packages from PyPI
pip install outlines langchain langgraph
```

---

## 🧱 Build and Install `llama-cpp-python` Correctly

Force it to use the **already-built CUDA version** instead of the vendored default:

```bash
CMAKE_ARGS="-DLLAMA_CMAKE_CUDA=on -DLLAMA_CPP_USE_VENDORED=OFF" \
FORCE_CMAKE=1 \
pip install git+https://github.com/abetlen/llama-cpp-python.git
```

This ensures:
- Your global CUDA build is used
- You get full GPU acceleration

---

## 🧪 Test the Setup

```bash
python -c "from llama_cpp import Llama; print('✅ Using llama_cpp from:', Llama.__module__)"
```

You can also test embedding or LLM inference with a `.gguf` model. Make sure your logs show tensors being assigned to GPU (not CPU).

---

## 🚀 Summary

- You now have a globally built, CUDA-enabled version of `llama.cpp`
- All Conda environments can use it via `LLAMA_CPP_LIB`
- Python packages like `llama-cpp-python`, `outlines`, and `langchain` work seamlessly
- Environment was created in **stages** due to `libcublas` constraints
- No need to rebuild `llama.cpp` per project

This setup allows you to manage multiple projects using Conda while sharing the same performant backend across environments.

## 🧠 Llama.cpp Integration & Model Selection Guide

This project integrates [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) with local GGUF models using GPU acceleration (CUDA). Below are important notes on model selection, system requirements, and build compatibility.

---

### 🔍 Supported Models & VRAM Guidelines

| Model                             | Size | Use Case                          | VRAM Required | Notes                                               |
|----------------------------------|------|-----------------------------------|---------------|-----------------------------------------------------|
| **LLaMA 3 8B (Q8_0)**            | 8B   | High-quality completions          | ~6–8 GB       | May exceed 8GB GPUs — partial CPU fallback needed   |
| **LLaMA 3.2 3B Instruct (Q8_0)** | 3B   | Fast instruct-tuned inference     | ~3–4 GB       | ✅ Actively being tested — fast + good quality       |
| **Mistral 7B (Q4_K_M)**          | 7B   | Balanced performance              | ~4–5 GB       | Excellent blend of speed and quality                |
| **Phi-2 (Q4_0)**                 | 1.6B | Prototyping, fast responses       | ~2–3 GB       | Small but powerful for many tasks                   |
| **TinyLLaMA 1.1B**               | 1.1B | Lightweight edge deployment       | ~1–2 GB       | Ideal for experiments or constrained environments   |

> ℹ️ *Use quantized models (`.gguf`) to reduce memory usage — especially `Q4_K_M` or `Q5_K_M` for optimal quality/speed balance.*

---

### 💡 Tips

- 🧠 **GPU Layer Tuning:** If you get "out of memory" errors, reduce `n_gpu_layers` (e.g. 20 instead of 100).
- ⚠️ **LangChain Compatibility:** LangChain's `LlamaCpp` may throw errors if the model partially falls back to CPU.
- 🧪 **Test interactively:** Load models in Python to determine GPU viability:

```python
from llama_cpp import Llama
llm = Llama(model_path="path/to/model.gguf", n_gpu_layers=20)
```

---

### 📌 Current Status

- ✅ `nomic-embed-text-v1.5` (CPU 0.09s)
- ✅ `Meta-LLaMA-3.1-8B-Instruct` (partial GPU fallback > 10s)
- ✅ `LLaMA-3.2-3B-Instruct-GGUF` (GPU < 1.5s)
- ✅ `LangChain` + `llama-cpp-python` integration works (GPU < 1.5S)
- ❌ Full-GPU for 8B on 8GB cards is not viable

---
