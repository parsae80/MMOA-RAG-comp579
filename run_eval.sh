cd /root/paddlejob/workspace/env_run/gpu
bash kill_gpu.sh
bash stop.sh

cd /root/paddlejob/workspace/env_run/rag
python selector_and_generator_get_answer_batch.py

cd /root/paddlejob/workspace/env_run/gpu
bash gpu.sh