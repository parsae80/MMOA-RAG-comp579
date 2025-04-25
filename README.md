# 📄 Multi-Agent Reinforcement Learning for Financial Document Retrieval-Augmented Generation

This repository implements and extends the Multi-Module Joint Optimization Algorithm for Retrieval-Augmented Generation (MMOA-RAG) to the domain of **financial question answering (QA)**.  
Inspired by [Chen et al., 2025](https://arxiv.org/abs/2501.15228), we leverage **Multi-Agent Reinforcement Learning (MARL)** to jointly optimize all RAG modules—**query rewriter**, **document selector**, and **answer generator**—under a shared reward signal.

## 🔍 Project Highlights

- **Domain Adaptation to Finance**  
  Adapts the MMOA-RAG framework to FinQA and other financial reports, where numerical precision and evidence faithfulness are paramount.

- **Custom PPO with domain-aware rewards**
  Keeps the retriever frozen and fine-tunes the generator via Proximal Policy Optimisation using a finance-tailored reward (relative-numeric-error + brevity penalty),     
  delivering materially higher answer accuracy than a vanilla F₁ baseline.

## 📚 Datasets Used

- **FinQA** — 8,281 annotated QA pairs for financial reasoning  

## ⚙️ Project Setup (Windows)

### 1. Clone the Repository

```bash
git clone [https://github.com/parsae80/MMOA-RAG-comp579.git]
cd MMOA-RAG-financialQA
```

### 2. Set Up a Python Virtual Environment

Ensure Python 3.10 is installed.

```bash
python3.10 -m venv venv
venv\Scripts\activate # For Window OS
source venv/bin/activate # for unix based OS 
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Initial Scripts

**Step 1: Retrieve Top-k Documents**

```bash
python get_top_k.py
```

**Step 2: Format Training Data**

```bash
python qr_s_g_sft_data_alpaca.py
python get_ppo_data_alpaca.py
```

**Step 3: Run SFT and PPO Training**

```bash
bash run_sft.sh
bash run_mapo.sh
```

### 🧪 7. (Optional) Evaluate or Serve the Model

**To Evaluate:**

```bash
python evaluate_qr_s_g.py
```

**To Serve Retrieval via Flask API:**

```bash
bash run_server.sh
```

Then query via:

```
GET /search?query=What is the EBITDA?
```

---

## 🌐 Project Architecture

### 📂 Base Project Structure

```plaintext
.
├── data/                       # Contains datasets (e.g., FinQA, SEC filings, MarketBeat), retrieval corpora, and helper scripts
├── LLaMA-Factory/              # Core implementation of MMOA-RAG including training, evaluation, and reinforcement learning logic
├── models/                     # Stores pretrained or fine-tuned models, tokenizer files, and generation configs
├── venv/ *                     # Python virtual environment (auto-generated) — DO NOT MODIFY
├── evaluate_qr_s_g.py          # Evaluates the final performance of the rewriter, selector, and generator using F1, EM, accuracy, and retrieval quality.
├── flask_server.py             # Runs a Flask API endpoint /search to return top-k documents using FAISS + Contriever.
├── get_ppo_data_alpaca.py      # Generates Alpaca-format PPO data for training selector and generator using reinforcement learning.
├── get_top_k.py                # Runs FAISS-based top-k document retrieval for a dataset and saves it to *_top_k_docs.jsonl.
├── normalize_answers.py        # Cleans and normalizes predicted and gold answers for accurate evaluation.
├── normalize_text.py           # Handles detailed text standardization, replacing control chars, quotes, and symbols.
├── qr_s_g_sft_data_alpaca.py   # Prepares Alpaca-style SFT data for the query rewriter, document selector, and answer generator using retrieved and rewritten documents.
├── run_server.sh               # Launches 8 parallel Flask API servers (flask_server.py) on ports 8000–8007, each bound to a separate GPU.
├── README.md                   # Project documentation and usage instructions
├── requirements.txt *          # Pip-based dependency list — DO NOT MODIFY
├── .gitignore *                # Git tracking exclusions (e.g., venv/, __pycache__) — DO NOT MODIFY
├── environment.yml *           # Conda environment spec for setting up dependencies — DO NOT MODIFY

*** All files and folders with an asterisk (*) should not be modified during the development of this project. ***
```

### 📂 Folder Structure Structure

#### Data Folder

```plaintext
data/
├── ambigqa/                    # Dataset folder with raw QA pairs, top-k documents, and format mapping for AmbigQA
│   ├── dataset_info.json           # Field mapping for Alpaca-style training format (instruction/input/output)
│   ├── test_data.jsonl             # QA pairs for test-time evaluation (Used by *evaluate_qr_s_g.py*.)
│   ├── top_k_train.jsonl           # Precomputed top-k retrieved documents for each question in train.jsonl. (Used by *qr_s_g_sft_data_alpaca.py*, *get_ppo_data_alpaca.py*.)
│   ├── train.jsonl                 # Raw training QA pairs (Used by *get_top_k.py* to retrieve documents for training, and for SFT/PPO)
│   └── val_top_k_docs.jsonl        # Retrieved top-k documents for validation questions (Used by *evaluate_qr_s_g.py*.)
├── psgs_w100.tsv               # Full corpus of 100-token Wikipedia passages (used as retrieval base)
├── temp.py                     # Script to inspect alignment between FAISS vectors and text passages (optional)
└── wikipedia.contriever        # FAISS index file (built using Contriever encoder) for fast dense retrieval
```

#### LLaMA-Factory Folder

```plaintext
LLaMA-Factory/
├── assets/                    # Templates, prompts, and configuration files for various models
├── docker/                    # Dockerfile and environment setup for container-based training
├── evaluation/                # Scripts for scoring model outputs using F1, BLEU, Rouge, etc.
├── examples/                  # Predefined examples to showcase training config usage
├── scripts/                   # Shell scripts for preprocessing, training, evaluation (some adapted for PPO)
├── src/                       # Core training modules: SFT, PPO, RLHF, data preprocessing, and agent logic
├── tests/                     # Unit and integration tests
├── README.md                  # LLaMA-Factory documentation and feature overview
├── README_zh.md               # Chinese-language version of the README
├── run_mapo.sh                # Shell script to run multi-agent PPO training (MAPPO)
├── run_sft.sh                 # Shell script to run supervised fine-tuning for QR/S/G
├── run.sh                     # Generic launcher for custom training pipelines
├── requirements.txt           # Python dependency list for pip installs
└── setup.py                   # Python packaging metadata for editable installs (pip install -e .)
```

#### Models Folder
```plaintext
models/
└── tinyllama/
    ├── config.json               # Model architecture & hyperparameter definitions
    ├── eval_results.json         # Evaluation metrics after validation (F1, EM, loss, accuracy, etc.)
    ├── generation_config.json    # Generation settings for inference (e.g., max_length, special tokens)
    ├── README.md                 # Metadata and HuggingFace-style documentation about the pretrained TinyLLaMA model
    ├── special_tokens_map.json   # Mapping of special tokens (e.g., <s>, </s>, <unk>) for tokenizer alignment
    ├── tokenizer_config.json     # Tokenizer behavior, chat formatting template, max length, padding strategy
    ├── tokenizer.json            # The actual tokenizer vocabulary, rules, and post-processing definitions
    └── tokenizer.model           # The SentencePiece model used to tokenize input text (binary format)
```

---

## 🧠 Reference

Chen, Y., Yan, L., Sun, W., Ma, X., Zhang, Y., Wang, S., Yin, D., & Yang, Y. (2025).  
*Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning.*  
arXiv preprint: [arXiv:2501.15228](https://arxiv.org/abs/2501.15228)
- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [TinyLLaMA Model](https://huggingface.co/cashue/tiny-llama)
