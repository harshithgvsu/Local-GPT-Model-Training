# Local-GPT-Model-Training

## 🚀 Overview
This repository provides code and utilities for **training, fine-tuning, and running a local GPT-style language model**. It includes modules for tokenization, model architecture, dataset processing, training loops, text generation, and Retrieval-Augmented Generation (RAG).

The goal is to allow anyone to build and run a completely **local LLM** with no cloud dependencies.

---

## 🎯 Key Features

- **Fully local GPT-style model training**
- Custom **BPE tokenizer**
- Modular **GPT architecture**
- Training pipeline with checkpoints
- **Text generation** with prompt input
- **Retrieval-Augmented Generation (RAG)** support
- Web-based data ingestion module
- Modular design for easy extension

---

## 🛠️ Getting Started

### 🔧 Prerequisites
You need:

- Python 3.8+
- PyTorch installed
- Git installed

---

### 📦 Setup Instructions

```bash
git clone https://github.com/harshithgvsu/Local-GPT-Model-Training.git
cd Local-GPT-Model-Training

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

🧠 Training the Model

Add your training text files to the data/ folder.

Run:
```
python train.py --dataset data/ --epochs 10 --model_size "small"
```

You can customize arguments like:

--epochs

--model_size

--batch_size

--learning_rate

--seq_len

💬 Text Generation

To generate text after training:
  ```
  python generate.py --checkpoint checkpoints/model_last.pth --prompt "Hello!"
  ```

Supports temperature, top-k, top-p, and more.

🔍 Retrieval-Augmented Generation (RAG)

  Use RAG for context-aware answers:
  ```
  python rag_chat.py --index data/index.db --model_checkpoint checkpoints/model_last.pth
  ```

The pipeline:

  - Retrieve relevant documents (retriever.py)

  - Feed them + your prompt to the model

  - Produce a grounded answer

🌐 Web Retriever

  Fetch web data and store it in your knowledge base:
  ```
  python web_retriever.py --query "LLM training guide"
  ```
