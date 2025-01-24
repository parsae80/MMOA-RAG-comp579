cd /root/paddlejob/workspace/env_run/gpu
bash kill_gpu.sh
bash stop.sh

cd /root/paddlejob/workspace/env_run/rag/LLaMA-Factory
llamafactory-cli train examples/train_lora/llama3_lora_ppo.yaml

cd /root/paddlejob/workspace/env_run/gpu
bash gpu.sh