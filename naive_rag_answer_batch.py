import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import get_conversation_template
import time
import json
from normalize_answers import *
from collections import Counter
import os

def normalize_answer_final(answer):
    pre_answer = answer.split('\n\n')[-1].split('Answer: ')[-1].split('The answer is: ')[-1]
    final_answer = normalize_answer(pre_answer)
    return final_answer

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
        
        if normalized_ground_truth in normalized_prediction or normalized_prediction in normalized_ground_truth:
            final_metric["acc"] += 1.0

        if normalized_prediction == normalized_ground_truth:
            final_metric["em"] += 1.0

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

def get_response(messages):
    if tokenizer.chat_template is not None:
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).cuda()
    else:
        conv = get_conversation_template(generator_model_path)
        for message in messages:
            conv.append_message(message["role"], message["content"])
        conv.append_message("assistant", "")
        input_ids = tokenizer(conv.get_prompt(), return_tensors="pt").input_ids.cuda()
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        # prefix_allowed_tokens_fn=lambda batch_id, prev_ids: allowed_tokens,
    )
    outputs = tokenizer.decode(
        outputs[0, input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    return outputs

def batch_get_response(messages_list):
    # Create a list of input_ids for all messages in the batch
    input_ids_list = []
    for messages in messages_list:
        if tokenizer.chat_template is not None:
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).cuda()
        else:
            conv = get_conversation_template(generator_model_path)
            for message in messages:
                conv.append_message(message["role"], message["content"])
            conv.append_message("assistant", "")
            input_ids = tokenizer(conv.get_prompt(), return_tensors="pt").input_ids.cuda()
        
        input_ids_list.append(input_ids)

    # Pad input_ids to the same length and create a single tensor
    input_ids_list = [input_ids.squeeze(0) for input_ids in input_ids_list]  # 向量需要是一维的

    # 找出最长序列的长度
    max_length = max(input_ids.size(0) for input_ids in input_ids_list)
    # 手动左填充
    input_ids_padded = torch.stack([
        torch.cat([input_ids.new_full((max_length - input_ids.size(0),), tokenizer.eos_token_id), input_ids], dim=0)
        for input_ids in input_ids_list
    ], dim=0)

    # 创建 attention mask
    attention_masks = torch.stack([
        torch.cat([torch.zeros(max_length - input_ids.size(0), dtype=torch.long), torch.ones(input_ids.size(0), dtype=torch.long)], dim=0)
        for input_ids in input_ids_list
    ], dim=0).cuda()

    # Generate outputs for all inputs in the batch
    outputs = model.generate(
        input_ids=input_ids_padded,
        attention_mask=attention_masks,
        max_new_tokens=100,
        do_sample=False,
        # temperature=0.7,
        # top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode each generated output separately
    results = []
    
    # for i in range(outputs.shape[0]):
    #     padding_length = max_length - input_ids_list[i].size(0)  # 计算每个输入的填充长度
    #     output = tokenizer.decode(
    #         outputs[i, padding_length:],  # 从非填充部分开始解码
    #         skip_special_tokens=True,
    #     )
    #     results.append(output)

    for i in range(outputs.shape[0]):
        # 因为是左填充，所以生成的输出部分从填充长度位置开始
        output = tokenizer.decode(
            outputs[i, input_ids_padded[i].size(0):],
            skip_special_tokens=True,
        )
        results.append(output)
    
    # for res in results:
    #     print(res)

    return results

def get_prefix_role_prompt(question, top_k_docs):
    input_content = "Question is: {}\n".format(str(question))
    for doc_id in range(len(top_k_docs)):
        doc_content = top_k_docs[doc_id]#['content']
        input_content = input_content + "Document {}: {}\n".format(str(doc_id), str(doc_content))
    input_content = input_content + "\nNow, answer the Question: {}, based on the above Documents".format(str(question))

    return [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question based on the given documents. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."},
        {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question based on the corresponding documents. Please provide the question and the corresponding documents.'},
        {'role': 'user', 'content': input_content},
        {'role': 'assistant', 'content': "OK, I received the Question and the corresponding Documents."},
        # {'role': 'user', 'content': "Before giving you question and documents, I show you somes examples.\n\nQuestion: Who was the first person killed in a car accident\nAnswer: Bridget Driscoll\n\nQuestion: Are both The New Pornographers and Kings of Leon American rock bands?\nAnswer: no\n\nQuestion: What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?\nAnswer: 6.213 km long\n\nQuestion: Which was the first European country to abolish capital punishment?\nAnswer: Norway\n\nPlease answer the question briefly like above examples."},
        # {'role': 'assistant', 'content': 'Okay, please provide the question and the corresponding documents.'}
    ]

def get_post_role_prompt():

    return {'role': 'user', 'content': "Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer with the form of **answer** and nothing else."}

def get_messages(question, top_docs, top_k=5):
    messages = get_prefix_role_prompt(question, top_docs)
    messages.append(get_post_role_prompt())

    return messages

top_k_docs_path = '/root/paddlejob/workspace/env_run/rag/data/val_top_k_docs.jsonl'
generator_model_path = '/root/paddlejob/workspace/env_run/rag/LLaMA-Factory/models/llama3_lora_ppo_3epochs'
# generator_model_path = '/root/paddlejob/workspace/env_run/rag/LLaMA-Factory/saves/llama3-8b/full/sft/checkpoint-2472'
# generator_model_path = '/root/paddlejob/workspace/env_run/rag/models/LLM-Research/Meta-Llama-3-8B-Instruct'

# loading questions-answer-topk_docs data
print('*'*20)
print('loading questions-answer-topk_docs data')
start_time = time.time()
# 创建一个空列表来存放字典
top_k_list = []
# 打开文件以读取
with open(top_k_docs_path, 'r') as file:
    for line in file:
        # json.loads() 函数将每行的json字符串转化为字典
        top_k_list.append(json.loads(line))
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# loading generator model
print('*'*20)
print('loading generator model')
model = AutoModelForCausalLM.from_pretrained(
    generator_model_path,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()
model = model.to(torch.float16)
tokenizer = AutoTokenizer.from_pretrained(generator_model_path)
tokenizer.padding_side = "left"
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# top_k_list = top_k_list[:30]

# get answers
questions = [top_k_list[k]['question'] for k in range(len(top_k_list))]
golden_answers = [top_k_list[k]['answer'] for k in range(len(top_k_list))]
predict_answers = []

# generating
print('*'*20)
print('generating')
start_time = time.time()

messages_list = []
for i in range(len(top_k_list)):
    cur_que_ans_docs = top_k_list[i]
    question = cur_que_ans_docs['question']
    answer = cur_que_ans_docs['answer']
    top_docs = [doc['content'] for doc in cur_que_ans_docs['top_k_docs']]
    messages = get_messages(question=question, top_docs=top_docs, top_k=len(top_docs))
    messages_list.append(messages)
print("len(messages_list): {}".format(len(messages_list)))

batch_size = 1
for i in tqdm(range(0, len(messages_list), batch_size)):
    messages_batch = messages_list[i: i+batch_size]
    answers_batch = batch_get_response(messages_batch)
    # print(answers_batch)
    for answer in answers_batch:
        answer = answer.replace("*", "")
        predict_answers.append(answer)


save_results_path = "/root/paddlejob/workspace/env_run/rag/data/naive_rag/selector_and_generator"
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)
# 保存答案为jsonl
with open(save_results_path+'/sft_1epoch.jsonl', 'w') as file: # 
    for i in range(len(predict_answers)):
        temp_dic = {'question_id': str(i), 'question': str(questions[i]), 'predict_answer': str(predict_answers[i]), 'golden_answer': str(golden_answers[i])}
        file.write(json.dumps(temp_dic) + '\n')

end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# getting F1 score
print(compute_scores(predict_answers, golden_answers))

print('Finished!')