import json
import matplotlib.pyplot as plt

def load_data(data_path, batch_size):
    data_path += '/reward_generator_final.txt'
    # 初始化一个空列表来存储浮点数
    data = []
    with open(data_path, 'r') as file:
        for line in file:
            # 去除行末的换行符或其他空白符
            stripped_line = line.strip()
            # 将字符串转换为浮点数
            float_value = float(stripped_line)
            # 将浮点数添加到列表中
            data.append(float_value)

    average_data = average_in_batches(data, batch_size)
    smoothed_data = exponential_weighted_moving_average(average_data)

    return smoothed_data # exponential_weighted_moving_average(data)

# 计算指数加权移动平均
def exponential_weighted_moving_average(data, weight=0.925):
    smoothed_data = [data[0]]  # 初始值为第一个数据点
    for i in range(1, len(data)):
        new_value = weight * smoothed_data[-1] + (1 - weight) * data[i]
        smoothed_data.append(new_value)
    return smoothed_data

def average_in_batches(data, batch_size=64):
    """
    计算列表中每隔 batch_size 个元素的平均值，并返回新的列表。

    :param data: 原始数据列表
    :param batch_size: 每组的大小
    :return: 包含每组平均值的新列表
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    
    averaged_list = []
    
    for i in range(0, len(data), batch_size):
        # 获取当前批次的子列表
        batch = data[i:i + batch_size]
        # 计算当前批次的平均值
        batch_average = sum(batch) / len(batch)
        # 将平均值添加到结果列表中
        averaged_list.append(batch_average)
    
    return averaged_list

data_path_qr_s = '/root/paddlejob/workspace/env_run/rag/LLaMA-Factory/saves/llama3-8b/lora/ambigqa/qr_s/ppo_1'
data_path_qr_g = '/root/paddlejob/workspace/env_run/rag/LLaMA-Factory/saves/llama3-8b/lora/ambigqa/qr_g/ppo_1'
data_path_s_g = '/root/paddlejob/workspace/env_run/rag/LLaMA-Factory/saves/llama3-8b/lora/ambigqa/s_g/ppo_1'
data_path_qr_s_g = '/root/paddlejob/workspace/env_run/rag/LLaMA-Factory/saves/llama3-8b/lora/ambigqa/qr_selector_and_generator/ppo_1'

batch_size = 64*4
data_qr_s = load_data(data_path_qr_s, batch_size)
data_qr_g = load_data(data_path_qr_g, batch_size)
data_s_g = load_data(data_path_s_g, batch_size)
data_qr_s_g = load_data(data_path_qr_s_g, batch_size)

# plt.style.use('ggplot')
# # plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# 设置图形尺寸和清晰度 (dpi)
# plt.figure(figsize=(9, 6), dpi=100)
plt.figure(figsize=(9, 6))

plt.plot([int(i*batch_size*2) for i in range(len(data_qr_s))], data_qr_s, label='MMOA-RAG w/o G', color='#6baed6', linewidth=3)  # 中度蓝
plt.plot([int(i*batch_size*2) for i in range(len(data_qr_g))], data_qr_g, label='MMOA-RAG w/o S', color='#fd8d3c', linewidth=3)  # 中度橙
plt.plot([int(i*batch_size*2) for i in range(len(data_s_g))], data_s_g, label='MMOA-RAG w/o QR', color='#74c476', linewidth=3)  # 中度绿
plt.plot([int(i*batch_size*2) for i in range(len(data_qr_s_g))], data_qr_s_g, label='MMOA-RAG', color='#e377c2', linewidth=3)  # 中度紫

# 设置标题和轴标签
# plt.title('Ablation about Optimizing Different Agents Number on AmbigQA Dataset', fontsize=14)
plt.xlabel('# Training Samples', fontsize=15)
plt.ylabel('F1 Score', fontsize=15)

# 设置图例
plt.legend(fontsize=15)

# 设置坐标轴刻度标签的字体大小
# 设置特定的纵坐标刻度
plt.yticks([0.38, 0.42, 0.46, 0.50], fontsize=15)
# 设置特定的横坐标刻度
plt.xticks([0, 20000, 40000, 60000], fontsize=15)

# 设置虚线的网格
plt.grid(True, linestyle='--')

# 保存为 jpg 文件
# plt.savefig('ablation.jpg', format='jpg')
plt.savefig('ablation.pdf')

