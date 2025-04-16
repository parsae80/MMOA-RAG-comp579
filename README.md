# PPO Fine-Tuning of TinyLLaMA for Financial QA using LLaMA-Factory

This repository contains a PPO fine-tuning setup using the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for a Retrieval-Augmented Generation (RAG) system focused on financial question answering (FinQA).

---
## LLama factory Directory:
Directory Structure Overview
Here's a breakdown of the key components within the LLaMA-Factory directory:​

assets/: Contains static resources such as images and diagrams used in documentation.​

docker/: Includes Dockerfiles and related scripts for containerizing the environment, facilitating consistent deployment across different systems.​

evaluation/: Houses scripts and tools for assessing model performance, including metrics computation and result visualization.​
arXiv

examples/: Provides sample configurations and scripts demonstrating various use cases and workflows for model fine-tuning and evaluation.​

scripts/: Contains utility scripts for tasks such as data preprocessing, training orchestration, and model conversion.​

src/: The core source code directory, encompassing modules for model architecture, training routines, and integration with reinforcement learning components.​

tests/: Includes unit and integration tests to ensure code reliability and correctness.​
arXiv

README.md & README_zh.md: Provide comprehensive documentation in English and Chinese, respectively, detailing setup instructions, usage guidelines, and project objectives.​

requirements.txt: Lists Python dependencies required to run the project.​

setup.py: Facilitates package installation and distribution.​

run.sh: A shell script to initiate standard training or evaluation workflows.​

run_mappo.sh: Specifically designed to launch training using the Multi-Agent Proximal Policy Optimization (MAPPO) algorithm within the MMOA-RAG framework.​

run_sft.sh: Used to commence supervised fine-tuning (SFT) processes.​

🚀 Getting Started
To set up and utilize the LLaMA-Factory toolkit:​

Clone the Repository:

bash
Copy
Edit
git clone https://github.com/chenyiqun/MMOA-RAG.git
cd MMOA-RAG/LLaMA-Factory
Install Dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run Training or Evaluation:

For standard training/evaluation:

bash
Copy
Edit
bash run.sh
For MAPPO-based training:

bash
Copy
Edit
bash run_mappo.sh
For supervised fine-tuning:

bash
Copy
Edit
bash run_sft.sh
Ensure that you have the necessary datasets and configurations in place as specified in the examples/ directory.


## 🔧 Project Setup

### ✅ Environment
We use `conda` to manage dependencies. Below is a minimal `requirements.txt` (already generated separately) that should be installed in your virtual environment.

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
