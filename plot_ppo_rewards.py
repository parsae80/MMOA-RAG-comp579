import json
import matplotlib.pyplot as plt

# 计算指数加权移动平均
def exponential_weighted_moving_average(data, weight=0.85):
    smoothed_data = [data[0]]  # 初始值为第一个数据点
    for i in range(1, len(data)):
        new_value = weight * smoothed_data[-1] + (1 - weight) * data[i]
        smoothed_data.append(new_value)
    return smoothed_data

save_path = '/root/paddlejob/workspace/env_run/rag/LLaMA-Factory/saves/llama3-8b/lora/selector_and_generator/ppo_only_update_selector/trainer_log.jsonl'

rewards = []
# 打开文件以读取
with open(save_path, 'r') as file:
    for line in file:
        # json.loads() 函数将每行的json字符串转化为字典
        temp_row = json.loads(line)
        rewards.append(temp_row['reward'])

# 获取平滑后的数据
smoothed_rewards = exponential_weighted_moving_average(rewards)

# 创建一个新的图形
plt.figure()

# 绘制原始的折线图
plt.plot([x*256 for x in range(len(rewards))], rewards, marker='x', markersize=5, linestyle='--', color='blue', alpha=0.2, label='Original')

# 绘制平滑后的折线图
plt.plot([x*256 for x in range(len(rewards))], smoothed_rewards, marker='o', markersize=3, linestyle='-', color='blue', alpha=1.0, label='Smoothed')

plt.grid(linestyle='-.')

# 添加标题和标签
plt.title('Rewards / EM of final answers (PPO for selector+generator)')
plt.xlabel('Number of QA pair for training')
plt.ylabel('Rewards / EM')

# 显示图形
plt.savefig('rewards_ppo_limited_tokens.jpg')