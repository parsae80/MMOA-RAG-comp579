import json
import matplotlib.pyplot as plt


# 计算指数加权移动平均
def exponential_weighted_moving_average(data, weight=0.8):
    smoothed_data = [data[0]]  # 初始值为第一个数据点
    for i in range(1, len(data)):
        new_value = weight * smoothed_data[-1] + (1 - weight) * data[i]
        smoothed_data.append(new_value)
    return smoothed_data

def plot_and_savefig(data_path, title='reward'):
    # data = []
    # # 打开文件以读取
    # with open(data_path, 'r') as file:
    #     for line in file:
    #         # json.loads() 函数将每行的json字符串转化为字典
    #         temp_row = json.loads(line)
    #         data.append(temp_row[title])

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

    SCALE = 16
    
    # 假设 float_list 已经被读取并转换为浮点数列表
    # float_list = data[224:-3200]  # 示例数据
    float_list = data[:]

    # 定义组的大小
    group_size = 32*SCALE
    # 初始化一个空列表来存储每组的平均值
    averaged_list = []
    # 遍历 float_list，每次走 group_size 步长
    for i in range(0, len(float_list), group_size):
        # 取出当前组的元素
        group = float_list[i:i+group_size]
        # 计算当前组的平均值
        average = sum(group) / len(group)
        # 将平均值添加到结果列表中
        averaged_list.append(average)

    # 打印结果列表
    data = averaged_list[:]

    # 获取平滑后的数据
    smoothed_data = exponential_weighted_moving_average(data)

    # 创建一个新的图形
    plt.figure()

    # 绘制原始的折线图
    plt.plot([x*64*SCALE for x in range(len(data))], data, marker='x', markersize=5, linestyle='--', color='blue', alpha=0.2, label='Original')

    # 绘制平滑后的折线图
    plt.plot([x*64*SCALE for x in range(len(data))], smoothed_data, marker='o', markersize=3, linestyle='-', color='blue', alpha=1.0, label='Smoothed')

    plt.grid(linestyle='-.')

    # 添加标题和标签
    plt.title(title + ' of PPO')
    plt.xlabel('Number of QA pair for training')
    plt.ylabel(title)

    # print((len(data)-1)*64*SCALE)

    # 显示图形
    plt.savefig('ppo_limited_tokens_{}.jpg'.format(title))


save_path = '/root/paddlejob/workspace/env_run/rag/LLaMA-Factory/saves/llama3-8b/lora/2wikimultihopqa/qr_g/ppo_1'
plot_and_savefig(save_path+'/reward_generator_final.txt', title='reward_generator_final')
plot_and_savefig(save_path+'/reward_selctor_aindoc.txt', title='reward_selctor_aindoc')