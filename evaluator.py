import json
from normalize_answers import *
from collections import Counter
import re

def normalize_answer_final(answer):
    pre_answer = answer.split('\n\n')[-1].split('Answer: ')[-1].split('The answer is: ')[-1]
    final_answer = normalize_answer(pre_answer)
    return final_answer

def preprocess_string(s):
    # 去除标点符号，转为小写，去除多余空白
    s = re.sub(r'[^\w\s]', '', s)  # 移除标点符号
    s = s.lower()  # 转为小写
    s = s.strip()  # 去除前后空白
    s = re.sub(r'\s+', ' ', s)  # 替换多个空白字符为单个空格
    return s

def exact_match(normalized_prediction, normalized_ground_truth):
    # 预处理字符串以标准化内容
    prediction = preprocess_string(normalized_prediction)
    ground_truth = preprocess_string(normalized_ground_truth)
    
    # 将字符串拆分为单词集合
    prediction_set = set(prediction.split())
    ground_truth_set = set(ground_truth.split())
    
    # 检查两个集合是否相等
    return 1.0 if prediction_set == ground_truth_set else 0.0

def calculate_accuracy(normalized_prediction, normalized_ground_truth):
    # 预处理字符串以标准化内容
    prediction = preprocess_string(normalized_prediction)
    ground_truth = preprocess_string(normalized_ground_truth)
    
    # 将字符串拆分为单词集合
    prediction_set = set(prediction.split())
    ground_truth_set = set(ground_truth.split())
    
    # 检查预测集合是否为标准集合的子集
    return 1.0 if prediction_set.issubset(ground_truth_set) else 0.0

def compute_scores(predict_answers, golden_answers):
    assert len(predict_answers) == len(golden_answers), "预测答案和标准答案的长度不相等"
    final_metric = {"acc": 0, "em": 0, "f1": 0, "precision": 0, "recall": 0}
    total = len(predict_answers)

    for prediction, ground_truth in zip(predict_answers, golden_answers):
        normalized_prediction = normalize_answer_final(prediction)
        normalized_ground_truth = normalize_answer_final(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        
        if normalized_ground_truth in normalized_prediction:# or normalized_prediction in normalized_ground_truth:
            final_metric["acc"] += 1.0
            
        if normalized_prediction == normalized_ground_truth:
            final_metric["em"] += 1.0

        # final_metric["acc"] +=  calculate_accuracy(normalized_prediction, normalized_ground_truth)
        # final_metric["em"] += exact_match(normalized_prediction, normalized_ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        final_metric["f1"] += f1
        final_metric["precision"] += precision
        final_metric["recall"] += recall

    for k in ['acc', 'em', 'f1', 'precision', 'recall']:
        final_metric[k] /= total

    return final_metric


save_path = '/root/paddlejob/workspace/env_run/rag/data/2wikimultihopqa/RRR.jsonl'

top_k_list = []
# 打开文件以读取
with open(save_path, 'r') as file:
    for line in file:
        # json.loads() 函数将每行的json字符串转化为字典
        top_k_list.append(json.loads(line))

predict_answers = [item['predict_answer'] for item in top_k_list]
golden_answers = [item['golden_answer'] for item in top_k_list]

# for i in range(100):
#     print(predict_answers[i], ' || ', golden_answers[i])

print(compute_scores(predict_answers, golden_answers))