Below is the updated README.md file  you can download or copy into your repo.

```markdown
# Multi-Agent RAG with PPO

This repository implements a multi-agent Retrieval-Augmented Generation (RAG) system using **Proximal Policy Optimization (PPO)** on top of a pretrained LLM (e.g., Llama-3-8B-Instruct). The goal is to optimize multiple modules (e.g., Query Rewriter, Document Selector, Answer Generator) cooperatively for question-answering tasks, possibly extended to specialized domains like **financial QA**.

---

## Features
1. **Supervised Fine-Tuning (SFT):** Warm-start each module (Rewriter, Selector, Generator) on domain-specific or general QA pairs.  
2. **Multi-Agent PPO:** Jointly optimize the modules with a shared reward signal (e.g., F1 score).  
3. **Flexible Retrieval Pipeline:** Easily integrate different document indexes (e.g., financial reports, general corpora).  
4. **Domain-Specific Rewards and Penalties:** Optionally add specialized constraints (e.g., numeric correctness checks for financial QA).

---

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/multi-agent-rag-ppo.git
cd multi-agent-rag-ppo
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Make sure you have a compatible GPU (e.g., A800, A100, or consumer GPU) and CUDA drivers set up.

### 3. Download or Prepare Dataset
- **Option A (General QA)**: Prepare or download the dataset used by the original approach (e.g., 2WikiMultihopQA).  
- **Option B (Financial QA)**: Gather your financial QA data (e.g., FinQA) and index any relevant documents (e.g., 10-K/10-Q reports).

### 4. Configure Environment Variables or YAML Settings (Optional)
- In `config/` or a `.env` file, specify hyperparameters, data paths, or model paths (e.g., `LLAMA_MODEL_PATH=/path/to/Llama-3-8B-Instruct`).

---

## Usage

### 1. Supervised Fine-Tuning (SFT)
To warm-start your modules:
```bash
python sft_train.py \
    --model_path ${LLAMA_MODEL_PATH} \
    --train_file data/finqa_train.json \
    --eval_file data/finqa_dev.json \
    --output_dir checkpoints/sft
```
- **Description**: This script fine-tunes the Query Rewriter, Selector, and Generator on your domain-specific data, giving them a better initialization than random weights.

### 2. PPO Training
Next, apply multi-agent PPO (or single-agent PPO if you choose). Example:
```bash
python ppo_train.py \
    --actor_init checkpoints/sft \
    --critic_init checkpoints/sft \
    --train_file data/finqa_train.json \
    --eval_file data/finqa_dev.json \
    --reward_metric "f1" \
    --num_steps 20000 \
    --batch_size 4 \
    --output_dir checkpoints/ppo
```
- **Key Flags**:
  - `--actor_init`: Warm-start the Actor from the SFT checkpoint.
  - `--critic_init`: Initialize Critic from either the same or another checkpoint.
  - `--reward_metric`: The metric used to compute the main reward (e.g., F1).
  - `--batch_size`: The number of rollout episodes per update (tweak for GPU memory).

### 3. Evaluation
After training, evaluate the final policy:
```bash
python evaluate.py \
    --model_path checkpoints/ppo \
    --test_file data/finqa_test.json \
    --output_file results/ppo_predictions.json
```
- **Description**: The script calculates final QA metrics (F1, exact match, etc.) and stores detailed logs.

---

## Financial QA Adaptation
To tailor the system specifically for financial QA:
1. **Use a Finance-Specific Retrieval Index**: For example, chunked 10-K/10-Q documents indexed by section.  
2. **Add Numeric Consistency Checks**: Consider penalizing the model if it references contradictory numbers.  
3. **Domain-Specific Penalties**: Encourage the Selector to focus on relevant statements; penalize the Rewriter for removing critical financial terms (like “EPS”).

---

## Troubleshooting & Tips
1. **Hardware**: PPO can be GPU-intensive. If you see out-of-memory errors, reduce `--batch_size` or switch to half precision (FP16).  
2. **Debug Mode**: Use a tiny subset of data (e.g., 100 samples) to confirm end-to-end functionality before full training.  
3. **Hyperparameter Tuning**: Adjust the learning rate, number of PPO epochs, or the clipping range (`--clip_range`) if training becomes unstable or converges too slowly.



## License
This project is licensed under the MIT License.
```

Simply copy-and-paste the above into your `README.md` file, and you’re set!
