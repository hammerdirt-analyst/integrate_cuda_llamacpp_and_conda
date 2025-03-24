"""
test_outlines_integration.py
author: roger erismann

llama.cpp + CUDA Functional Test Script with Model & Embedding Validation
Test Script for Outlines + Transformers Integration (Local Setup)

This script evaluates whether the Outlines library can run successfully
using locally cached Hugging Face Transformers models within the current
development environment.

Tests include:
- Classification using `generate.choice` (e.g., URGENT vs STANDARD)
- Classification using `generate.json` with a Pydantic schema
- Named entity extraction using structured output (e.g., pizza order details)

The script reports:
- Elapsed time for each test
- Model used
- Inference results

This helps validate compatibility between Outlines, the Phi-3-mini model, and
hardware/conda setup without relying on cloud inference or APIs.
"""

import os
import time
from enum import Enum
from pydantic import BaseModel
from outlines import models, generate
from outlines.samplers import BeamSearchSampler, MultinomialSampler
from transformers import GenerationConfig
from jinja2 import Template

# === Config ===
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
cache_dir = "/home/roger/.cache/huggingface/hub"
transformer_model_id = "microsoft/Phi-3-mini-4k-instruct"
modelx = models.transformers(transformer_model_id)

# Shared input
requests = [
    'The sky is blue',
    'I can not breathe',
    'Red three one y x',
    'Help me',
    'this really hurts',
    'Gu si to skna sif sso',
    "My hair is on fire! Please help me!!!",
    "Just wanted to say hi"
]

# === Helpers ===
def run_test(label, fn):
    print(f"\n===== {label.upper()} (Model: {transformer_model_id}) =====")
    start = time.time()
    fn()
    elapsed = time.time() - start
    print(f"[TIME] Elapsed: {elapsed:.2f}s")
    print("=" * 60)

def render_prompts(template_path, context_list, var_name):
    with open(template_path) as f:
        template = Template(f.read())
    return [template.render(**{var_name: val}) for val in context_list]

# === Tests ===
def classification_choice():
    prompts = render_prompts("templates/customer_support.jinja", requests, "request")
    generator = generate.choice(modelx, ["URGENT", "STANDARD"], sampler=BeamSearchSampler())
    labels = generator(prompts)
    for req, label in zip(requests, labels):
        print(f"{label:>8} ← {req}")

def classification_json():
    class Label(str, Enum):
        urgent = "URGENT"
        standard = "STANDARD"
    class Classification(BaseModel):
        label: Label

    prompts = render_prompts("templates/customer_support.jinja", requests, "request")
    generator = generate.json(modelx, Classification, sampler=MultinomialSampler())
    labels = generator(prompts)
    for req, label in zip(requests, labels):
        print(f"{label.label:>8} ← {req}")

def order_extraction():
    class Pizza(str, Enum):
        margherita = "Margherita"
        pepperonni = "Pepperoni"
        calzone = "Calzone"
    class Order(BaseModel):
        pizza: Pizza
        number: int

    orders = [
        "Hi! I would like to order two pepperonni pizzas and would like them in 30mins.",
        "Is it possible to get 12 margheritas?"
    ]
    prompts = render_prompts("templates/take_order.jinja", orders, "order")
    generator = generate.json(modelx, Order)
    results = generator(prompts)
    for o, r in zip(orders, results):
        print(f"→ Order: {r.number} x {r.pizza} (from: {o})")

# === Run Tests ===
run_test("Classification (choice)", classification_choice)
run_test("Classification (JSON)", classification_json)
run_test("Named Entity Extraction (Order)", order_extraction)
