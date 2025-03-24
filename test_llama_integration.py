"""
test_llama_integration.py
author: roger erismann

llama.cpp + CUDA Functional Test Script with Model & Embedding Validation

This script verifies that `llama.cpp` is correctly integrated with CUDA support and functional
within the current environment. It loads one or more LLM models via `llama-cpp-python`, runs
basic inference to ensure model loading and response generation, and optionally checks various
embeddings and configurations.

Key features:
- Validates llama.cpp CUDA support (build and runtime)
- Loads one or more LLaMA-compatible models
- Runs minimal inference (e.g., system prompt or basic question-answering)
- Tests embeddings or tokenization if specified
- Designed to be a fast sanity check for CUDA-enabled LLM environments

Use this script to confirm that your llama.cpp setup works with different model sizes, backends,
and hardware acceleration (e.g., in Docker containers or fresh environments).
"""



import time
import os
from llama_cpp import Llama
from contextlib import redirect_stdout, redirect_stderr
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Define models
MODELS = [
    {
        "path": "models/Llama-3.2-3B-Instruct-Q6_K.gguf",
        "name": "llama-3.2-3B",
        "chat_format": "llama-3",
        "mode": "chat"
    },
     {
        "path": "models/nomic-embed-text-v1.5.Q8_0.gguf",
        "name": "nomic-embed",
        "chat_format": None,
        "mode": "embedding"
    }
]

def run_with_logging(model_name, func):
    log_path = os.path.join("logs", f"{model_name}.log")
    with open(log_path, "w") as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
        return func()

def test_llama_cpp_chat(cfg):
    start = time.time()

    def run():
        model = Llama(model_path=cfg["path"], n_gpu_layers=22)
        prompt = "Question: What is the capital of France? make your answer as short as possible\nAnswer:"
        output = model(prompt, max_tokens=128, temperature=0.2)
        return output["choices"][0]["text"].strip()

    try:
        response = run_with_logging(cfg['name'], run)
        elapsed = time.time() - start
        print(f"✅ {cfg['name']} completed in {elapsed:.2f}s")
        print(f"💬 Response: {response}")
    except Exception as e:
        print(f"❌ {cfg['name']} failed: {e}")

def test_llama_cpp_embedding(cfg):
    start = time.time()

    def run():
        model = Llama(model_path=cfg["path"], embedding=True, n_gpu_layers=22)
        return model.embed("hello world")

    try:
        embedding = run_with_logging(cfg['name'], run)
        elapsed = time.time() - start
        print(f"✅ {cfg['name']} completed in {elapsed:.2f}s")
        print(f"🧬 First 3 embedding values: {embedding[:3]}")
    except Exception as e:
        print(f"❌ {cfg['name']} failed: {e}")

def test_langchain_chat(cfg):
    start = time.time()

    def run():
        llm = LlamaCpp(
            model_path=cfg["path"],
            temperature=0.2,
            max_tokens=128,
            n_ctx=2048,
            n_gpu_layers=22
        )
        template = "Question: {question}\nAnswer:"
        prompt = PromptTemplate(template=template, input_variables=["question"])
        chain = LLMChain(prompt=prompt, llm=llm)
        return chain.run("What is the capital of France? make your answer as short as possible")

    try:
        response = run_with_logging(cfg['name'] + "-langchain", run)
        elapsed = time.time() - start
        print(f"✅ LangChain {cfg['name']} completed in {elapsed:.2f}s")
        print(f"💬 Response: {response}")
    except Exception as e:
        print(f"❌ LangChain {cfg['name']} failed: {e}")

def main():
    for cfg in MODELS:
        if cfg["mode"] == "chat":
            test_llama_cpp_chat(cfg)
        elif cfg["mode"] == "embedding":
            test_llama_cpp_embedding(cfg)

    for cfg in MODELS:
        if cfg["mode"] == "chat":
            test_langchain_chat(cfg)

if __name__ == "__main__":
    main()