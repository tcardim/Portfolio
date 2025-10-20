# 🧠 Application of Parameter-Efficient Fine-Tuning (PEFT) using LoRA

## 📘 Overview
This project demonstrates how to apply **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA — Low-Rank Adaptation** to improve a pre-trained **DistilBERT** model on IMDb movie review sentiment classification.  
The goal is to show that meaningful performance gains can be achieved with a fraction of the parameters trained, reducing compute cost while maintaining accuracy.

---

## 🎯 Objectives
- Load and evaluate a pre-trained LLM (`distilbert-base-uncased-finetuned-sst-2-english`) on IMDb sentiment data.  
- Apply **LoRA adapters** for efficient fine-tuning on a small subset of the dataset.  
- Compare **baseline vs. fine-tuned** model performance using accuracy and F1-score metrics.

---

## 🧩 Methodology
1. **Dataset** – Loaded IMDb sentiment dataset (25 K train / 25 K test) from Hugging Face and shuffled data to avoid label ordering bias.  
2. **Baseline Evaluation** – Measured initial performance of DistilBERT SST-2 on IMDb without modification.  
3. **Fine-Tuning Setup** – Wrapped the base model with LoRA adapters targeting the query & value projection layers; trained for 5 epochs on 5 K samples.  
4. **Training Optimization** – Used `fp16` mixed precision, 32-sample batches, and a 2e-4 learning rate for efficiency.  
5. **Evaluation** – Compared pre- and post-fine-tuning results via `accuracy` and `f1` metrics, observing +2 pp accuracy improvement.

---

## ⚙️ Environment & Dependencies
### Packages Used
```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
import numpy as np
from evaluate import load as load_metric
from transformers.utils import logging
import torch, os
from peft import LoraConfig, get_peft_model, PeftModel