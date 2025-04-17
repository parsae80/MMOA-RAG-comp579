# 📄 Multi-Agent Reinforcement Learning for Financial Document Retrieval-Augmented Generation

This repository implements and extends the Multi-Module Joint Optimization Algorithm for Retrieval-Augmented Generation (MMOA-RAG) to the domain of **financial question answering (QA)**.  
Inspired by [Chen et al., 2025](https://arxiv.org/abs/2501.15228), we leverage **Multi-Agent Reinforcement Learning (MARL)** to jointly optimize all RAG modules—**query rewriter**, **document selector**, and **answer generator**—under a shared reward signal.

## 🔍 Project Highlights

- **Domain Adaptation to Finance**  
  Applies MMOA-RAG to complex financial texts such as earnings call transcripts and SEC filings, where accurate retrieval and reasoning are critical.

- **Multi-Agent Proximal Policy Optimization (MAPPO)**  
  Each module is treated as a cooperating agent trained via **shared reward signals** (F1 score + retrieval accuracy), enabling joint policy optimization.

- **Ablation via Module Freezing**  
  Investigates how individual components (e.g., query rewriting) contribute to overall performance by freezing them during training.

- **Baseline Comparisons**  
  Evaluated against:
  - Standard RAG
  - Single-agent PPO

## 📚 Datasets Used

- **FinQA** — 8,281 annotated QA pairs for financial reasoning  
- **Earnings Call Transcripts** — Extracted from MarketBeat  
- **SEC Filings** — 10-K and 10-Q reports from the EDGAR database

## ⚙️ Project Setup (Windows)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/MMOA-RAG-financialQA.git
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

## ✅ TODO

### 📌 PHASE 1 — Dataset & Corpus Preparation

#### 🔹 Step 1: Prepare QA dataset
- [ ] Create `train.jsonl` and `test_data.jsonl`
  - Each line should be formatted as:
    ```json
    {"question": "What is the net income?", "answer": "2.3 billion"}
    ```
  - Place these files in `data/your_dataset_name/`.

#### 🔹 Step 2: Build retrieval corpus
- [ ] Chunk your financial documents (e.g., SEC filings, earnings calls) into ~100-token passages.
- [ ] Save the corpus in `psgs_w100.tsv` format:
    ```tsv
    <Title> \t <Passage Text>
    ```

#### 🔹 Step 3: Create FAISS index with Contriever
- [ ] Use `facebook/contriever` to encode all passages.
- [ ] Build a FAISS index and save it as `wikipedia.contriever`.
- [ ] Store both `.tsv` and `.contriever` files in the `data/` directory.

---

### 📌 PHASE 2 — Top-k Document Retrieval

#### 🔹 Step 4: Generate top-k document lists
- [ ] Run the following:
    ```bash
    python get_top_k.py
    ```
- [ ] This will generate:
  - `top_k_train.jsonl`
  - `val_top_k_docs.jsonl`
- [ ] Store both in `data/your_dataset_name/`.

---

### 📌 PHASE 3 — Supervised Fine-Tuning (SFT)

#### 🔹 Step 5: Format SFT data
- [ ] Run:
    ```bash
    python qr_s_g_sft_data_alpaca.py
    ```
- [ ] Ensure `dataset_info.json` is present to map fields to Alpaca format (`instruction`, `input`, `output`).

#### 🔹 Step 6: Fine-tune modules
- [ ] Edit and run:
    ```bash
    bash run_sft.sh
    ```
- [ ] This will fine-tune the Query Rewriter, Document Selector, and Answer Generator modules using SFT.

---

### 📌 PHASE 4 — PPO Training (MAPPO)

#### 🔹 Step 7: Format PPO data
- [ ] Run:
    ```bash
    python get_ppo_data_alpaca.py
    ```

#### 🔹 Step 8: Train with MAPPO
- [ ] Edit and run:
    ```bash
    bash run_mapo.sh
    ```
- [ ] This trains QR, Selector, and Generator **jointly** using Multi-Agent PPO and a shared reward signal.

---

### 📌 PHASE 5 — Model Evaluation

#### 🔹 Step 9: Evaluate performance
- [ ] Run:
    ```bash
    python evaluate_qr_s_g.py
    ```
- [ ] Input files:
  - `val_top_k_docs.jsonl`
  - `test_data.jsonl`
- [ ] Output: Model predictions, F1 score, Exact Match (EM), and accuracy.

---

### 📌 PHASE 6 — Module Freezing & Ablation (Optional)

#### 🔹 Step 10: Run ablation studies
- [ ] Freeze the Query Rewriter → measure impact on retrieval and generation.
- [ ] Freeze the Document Selector → use random docs and measure degradation.
- [ ] Compare overall model behavior with and without specific modules.

---

### 📌 PHASE 7 — Interactive Flask API (Optional)

#### 🔹 Step 11: Launch document search server
- [ ] Run:
    ```bash
    bash run_server.sh
    ```
- [ ] Query the API endpoint:
    ```
    GET /search?query=What is the EBITDA?
    ```

---

### ✅ Summary Pipeline

```plaintext
Raw QA + Financial Docs
   ↓
psgs_w100.tsv + wikipedia.contriever
   ↓
get_top_k.py → top_k_train.jsonl + val_top_k_docs.jsonl
   ↓
qr_s_g_sft_data_alpaca.py → SFT dataset → run_sft.sh
   ↓
get_ppo_data_alpaca.py → PPO dataset → run_mapo.sh
   ↓
evaluate_qr_s_g.py → final results
```

---

## 🧠 Reference

Chen, Y., Yan, L., Sun, W., Ma, X., Zhang, Y., Wang, S., Yin, D., & Yang, Y. (2025).  
*Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning.*  
arXiv preprint: [arXiv:2501.15228](https://arxiv.org/abs/2501.15228)
- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [TinyLLaMA Model](https://huggingface.co/cashue/tiny-llama)
