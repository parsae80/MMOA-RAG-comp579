# PPO Fine-Tuning of TinyLLaMA for Financial QA using LLaMA-Factory

This repository contains a PPO fine-tuning setup using the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for a Retrieval-Augmented Generation (RAG) system focused on financial question answering (FinQA).

---
If you dont have linux, here is the setup for you:
## Setup
# 🐧 Setting Up WSL + Ubuntu for LLaMA-Factory Projects

This guide walks you through installing **Windows Subsystem for Linux (WSL)**, setting up **Ubuntu**, and preparing your system with **Miniconda** for Python-based machine learning projects like **LLaMA-Factory**.

---

## 🚀 Step 1: Install WSL and Ubuntu

1. Open **PowerShell as Administrator**.
2. Run the following command to install WSL and Ubuntu:

```powershell
wsl --install
```

This will install:

WSL (Windows Subsystem for Linux)

Ubuntu (default: Ubuntu 22.04)

Required virtualization features

Restart your system when prompted.

🖥 Step 2: Launch Ubuntu
Once your system reboots:

Go to the Start Menu and search for "Ubuntu".

Launch Ubuntu. On the first run:

Set your Linux username and password.

It will configure your environment.

You are now inside the Ubuntu shell on Windows 🎉

🛠 Step 3: Update Ubuntu and Install Dev Tools
In the Ubuntu terminal:

bash
Copy
Edit
sudo apt update && sudo apt upgrade -y
sudo apt install -y git build-essential curl wget zip unzip
This ensures your system has all necessary tools for Python and compilation.

🐍 Step 4: Install Miniconda (Python Environment Manager)
Download Miniconda:

bash
Copy
Edit
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
Install it:

bash
Copy
Edit
bash Miniconda3-latest-Linux-x86_64.sh
Follow the prompts:

Accept the license.

Install to the suggested path (or your preferred one).

Confirm yes to initialize Conda.

Activate Conda:

bash
Copy
Edit
source ~/.bashrc
Create and activate your project environment:

bash
Copy
Edit
conda create -n rag_env python=3.10 -y
conda activate rag_env
You now have a Python 3.10 environment ready for your project

## 🔧 Project Setup

### ✅ Environment

Run the following in your terminal to setup for the project.

```sh
python3.10 -m venv venv # Python 3.10 is required as many dependencies only work with v3.10
venv\Scripts\activate # run this for Window OS
source venv/bin/activate # run this for Unix based OS
pip install -r requirements.txt
```

### ✅ Model
- **Base Model**: [`TinyLLaMA`](https://huggingface.co/cashue/tiny-llama) (saved locally)
- **Path**: `MMOA-RAG-comp579/models/tinyllama`

### ✅ Dataset
A very small JSONL dataset was used to test the pipeline:
```json
{"instruction": "Where was Albert Einstein born?", "input": "", "output": "Ulm, Germany"}
{"instruction": "What is the capital of Canada?", "input": "", "output": "Ottawa"}
{"instruction": "How many moons does Mars have?", "input": "", "output": "2"}
```
- **Format**: Alpaca-style (`instruction`, `input`, `output`)
- **Location**: `MMOA-RAG-comp579/data/ambigqa/train_data.jsonl`

### ✅ Reward Function
- We used a built-in `get_rewards` method using **F1-score** based on predicted vs gold answers.
- No separate reward model was loaded (direct metric-based reward only).

---

## 🚀 Training Run

### PPO Command Used:
```bash
PYTHONPATH=./src torchrun --nproc_per_node=1 src/llamafactory/launcher.py \
  --stage ppo \
  --do_train \
  --model_name_or_path /MMOA-RAG-comp579/models/tinyllama \
  --dataset ambigqa \
  --dataset_dir /MMOA-RAG-comp579/data \
  --template alpaca \
  --finetuning_type full \
  --reward_model /MMOA-RAG-comp579/models/tinyllama \
  --reward_model_type full \
  --output_dir /MMOA-RAG-comp579/llama-outputs/ppo_train \
  --overwrite_cache \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --save_steps 10 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --plot_loss \
  --report_to none
```
> ⚠️ Note: The dataset only had 3 samples — this was just to verify that PPO fine-tuning could execute without error.

---

---

## 📌 TODO
- Replace toy dataset with full FinQA or HotpotQA-style dataset
- Extend `get_rewards()` to cover document relevance and generator penalization
- Use multi-GPU setup for scaling
- Add evaluation scripts

---

## 📎 References
- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [TinyLLaMA Model](https://huggingface.co/cashue/tiny-llama)
