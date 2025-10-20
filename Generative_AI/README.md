# 🎓 Udacity Generative AI Nanodegree

## 🧠 Overview
This repository contains my completed projects and learning artifacts from **Udacity’s Generative AI Nanodegree** program.  
The coursework focuses on building a deep understanding of **Large Language Models (LLMs)**, **parameter-efficient fine-tuning (PEFT)**, **transfer learning**, and **prompt engineering**, culminating in hands-on projects applying these techniques using modern frameworks like **Hugging Face Transformers** and **PyTorch**.

The projects here demonstrate not only model implementation but also **evaluation, optimization, and interpretability** — reflecting both the theoretical and practical sides of generative AI development.

---

## 📚 Nanodegree Structure

| Module | Focus Area | Core Skills |
|:-------|:------------|:------------|
| **1. Generative AI Fundamentals** | Understanding foundation models and transformers | LLM architecture, embeddings, tokenization, attention mechanisms |
| **2. Image Generation with Diffusion Models** | Building and fine-tuning image generation models | Denoising diffusion, latent space, Stable Diffusion |
| **3. Parameter-Efficient Fine-Tuning (PEFT)** | Adapting pre-trained LLMs efficiently | LoRA, adapters, fine-tuning pipelines |
| **4. Prompt Engineering & Applications** | Leveraging LLMs through controlled prompting | Few-shot learning, instruction tuning, prompt design |
| **5. Capstone Project** | End-to-end generative AI system | LLM integration, evaluation, deployment considerations |

---

## 🧩 Projects

### **1️⃣ PEFT Sentiment Analysis using LoRA**
📁 [`01_Peft_LoRA_Sentiment_Analysis`](./01_Peft_LoRA_Sentiment_Analysis)

**Goal:**  
Fine-tune a pre-trained DistilBERT model using **LoRA** adapters to improve sentiment classification performance on the IMDb movie review dataset.

**Highlights:**
- Demonstrates **parameter-efficient fine-tuning (PEFT)** concepts.  
- Achieved **+2% accuracy improvement** (89% → 91%) using <10% of parameters.  
- Showcases practical LoRA configuration and training pipeline with Hugging Face.

**Skills:** LoRA, Hugging Face Transformers, PyTorch, Datasets, PEFT, Model Evaluation

---

### **2️⃣ Image Generation with Stable Diffusion**
📁 [`02_Image_Generation_Diffusion`](./02_Image_Generation_Diffusion)

**Goal:**  
Train and customize a **diffusion-based image generator** using latent denoising and fine-tuning on a small dataset of concept images.

**Highlights:**
- Implements a **Denoising Diffusion Probabilistic Model (DDPM)** pipeline.  
- Visualizes progressive denoising across inference steps.  
- Demonstrates fine-tuning for **concept style transfer** (custom image theme adaptation).

**Skills:** Diffusion Models, Denoising Autoencoders, PyTorch, Latent Representations, Image-to-Image Generation

---

### **3️⃣ Prompt Engineering & Text Generation**
📁 [`03_Prompt_Engineering_TextGen`](./03_Prompt_Engineering_TextGen)

**Goal:**  
Design and evaluate effective **prompt templates** for GPT-style models to perform tasks like summarization, sentiment classification, and creative writing.

**Highlights:**
- Experiments with **few-shot** and **chain-of-thought** prompting.  
- Evaluates response quality via **BLEU**, **ROUGE**, and **semantic similarity**.  
- Showcases prompt tuning for deterministic vs. creative responses.

**Skills:** Prompt Engineering, LLM Evaluation, Few-Shot Learning, NLP Metrics, OpenAI API / Hugging Face Inference

---

### **4️⃣ Capstone – Generative AI Product Prototype**
📁 [`04_Capstone_Generative_Product`](./04_Capstone_Generative_Product)

**Goal:**  
Develop a working prototype of a **generative AI application** combining LLM reasoning and retrieval-augmented generation (RAG).  
For example, a “**Smart Review Assistant**” that summarizes and classifies user feedback while generating improvement suggestions.

**Highlights:**
- Combines **PEFT-tuned model + vector retrieval** (FAISS).  
- Uses **LangChain / Hugging Face pipelines** for RAG orchestration.  
- Integrates evaluation metrics for both generative quality and factual accuracy.

**Skills:** RAG, LangChain, FAISS, Vector Databases, LLM Evaluation, Streamlit / Gradio Interface

---

## 🧰 Tools & Frameworks
- **Python 3.10+**
- **PyTorch**
- **Hugging Face Transformers**
- **PEFT / LoRA**
- **Diffusers**
- **LangChain**
- **Evaluate / Datasets**
- **Jupyter / nbconvert**

---

## 🧠 Key Learnings
- Fine-tuning efficiency: small adapters can unlock major improvements.  
- Model control through prompting often rivals retraining in flexibility.  
- Diffusion models rely on intuitive but powerful latent-space manipulations.  
- Combining retrieval with generation (RAG) enhances factual grounding.  
- Generative AI is as much about **evaluation and alignment** as it is about creativity.

---

## 📜 References
- [Udacity Generative AI Nanodegree](https://www.udacity.com/course/generative-ai--nd600)  
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers/index)  
- [LoRA: Low-Rank Adaptation of LLMs (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)  
- [Diffusers Library](https://huggingface.co/docs/diffusers/index)

---

👤 **Author:** [Thomas Orgler](https://github.com/tcardim)  
🎓 *Udacity Generative AI Nanodegree Portfolio — 2025*
