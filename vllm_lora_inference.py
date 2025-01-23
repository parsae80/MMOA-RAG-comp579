from vllm import LLM, SamplingParams

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
import random
from transformers import LogitsProcessorList, LogitsProcessor
from limited_tokens import get_allowed_tokens, AllowedTokensLogitsProcessor


def normalize_answer_final(answer):
    pre_answer = answer.split('\n\n')[-1].split('Answer: ')[-1].split('The answer is: ')[-1]
    final_answer = normalize_answer(pre_answer)
    return final_answer

def convert_to_int_list(s, K_candidate):
    int_list = []
    for x in s.split(','):
        try:
            # 尝试转换为整数
            num = int(x.strip())
            if num < 0 or num >= K_candidate:
                num = random.randint(0, K_candidate-1)
                print('模型输出的文档ID为 {} , 这不在符合规定的范围内, 重新随机赋值为: {}\n'.format(int(x.strip()), num))
            int_list.append(num)
        except ValueError:
            num = random.randint(0, K_candidate-1)
            print('转换原始字符 {} 失败, 随机赋值为: {}'.format(x, num))
            print('对应完整的原始answers为:', s, '\n')
            continue
    return int_list
    
def extract_digits(input_string):
    input_string = input_string.replace(",", "")
    # 创建一个空列表来存储结果
    digits_list = []
    
    # 遍历输入字符串中的每个字符
    for char in input_string:
        # 检查字符是否为数字
        if char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            # 将字符转换为整数并添加到列表中
            digits_list.append(int(char))

    # 去重复
    my_list = digits_list
    unique_list = []
    for item in my_list:
        if item not in unique_list:
            unique_list.append(item)
    
    return unique_list

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

def batch_get_response(model, tokenizer, messages_list, logits_processor=[]):
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
    if len(logits_processor) != 0:
        outputs = model.generate(
            input_ids=input_ids_padded,
            attention_mask=attention_masks,
            max_new_tokens=19,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor,
        )
    else:
        outputs = model.generate(
            input_ids=input_ids_padded,
            attention_mask=attention_masks,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode each generated output separately
    results = []
    for i in range(outputs.shape[0]):
        # 因为是左填充，所以生成的输出部分从填充长度位置开始
        output = tokenizer.decode(
            outputs[i, input_ids_padded[i].size(0):],
            skip_special_tokens=True,
        )
        results.append(output)

    return results

def get_selector_prefix_role_prompt(question, top_k_docs):
    input_content = "Question is: {}\n".format(str(question))
    for doc_id in range(len(top_k_docs)):
        doc_content = top_k_docs[doc_id]#['content']
        input_content = input_content + "Document {}: {}\n".format(str(doc_id), str(doc_content))

    message = [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Your task is to output the ID of the candidate Documents (0,1,2,...,{}) which are helpful in answering the Question.".format(len(top_k_docs)-1)}, 
        {'role': 'assistant', 'content': 'Okay, I will provide the ID of candidate Documents which are helpful in answering the Question.'},
        {'role': 'user', 'content': input_content},
        {'role': 'assistant', 'content': "OK, I received the Question and the candidate Documents."}
    ]

    return message

def get_generator_prefix_role_prompt(question, top_k_docs):
    input_content = "Question is: {}\n".format(str(question))
    for doc_id in range(len(top_k_docs)):
        doc_content = top_k_docs[doc_id]#['content']
        input_content = input_content + "Document {}: {}\n".format(str(doc_id), str(doc_content))

    if len(top_k_docs) > 0:
        input_content = input_content + "\nNow, answer the Question: {}, based on the above Documents".format(str(question))
        message = [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question based on the given documents. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."},
        {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question based on the corresponding documents. Please provide the question and the corresponding documents.'},
        {'role': 'user', 'content': input_content},
        {'role': 'assistant', 'content': "OK, I received the Question and the corresponding Documents."}
    ]

    elif len(top_k_docs) == 0:
        input_content = input_content + "\nNow, answer the Question: {}.".format(str(question))
        message = [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."},
        {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question. Please provide the question.'},
        {'role': 'user', 'content': input_content},
        {'role': 'assistant', 'content': "OK, I received the Question."}
    ]

    return message

def get_selector_post_role_prompt(question, top_k_docs):

    return {'role': 'user', 'content': "Now, output the ID of the candidate Documents (0,1,2,...,{}) which are helpful in answering the Question: {}, for example, in the following format: i,j,k,l (i,j,k,l represent the concrete Documents ID 0,1,2,... {}). Do not output duplicate Document ID.".format(len(top_k_docs)-1, str(question), len(top_k_docs)-1)}

def get_generator_post_role_prompt(top_docs):

    if len(top_docs) > 0:
        message = {'role': 'user', 'content': "Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer with the form of **answer** and nothing else."}
    elif len(top_docs) == 0:
        message = {'role': 'user', 'content': "Given the Question, predict the answer to the Question as briefly and accurately as possible. Only give the brief and accurate answer with the form of **answer** and nothing else."}

    return message

def get_messages(question, top_docs):
    messages = get_generator_prefix_role_prompt(question, top_docs)
    messages.append(get_generator_post_role_prompt(top_docs))

    return messages

def get_selector_messages(question, top_docs):
    messages = get_selector_prefix_role_prompt(question, top_docs)
    messages.append(get_selector_post_role_prompt(question, top_docs))

    return messages

# prompts = [
#     "Hello, my name is",
#     # "The president of the United States is",
#     # "The capital of France is",
#     # "The future of AI is",
# ]
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# llm = LLM(model="/root/paddlejob/workspace/env_run/rag/LLaMA-Factory/models/llama3_lora_sft", gpu_memory_utilization=0.3)

# outputs = llm.generate(prompts, sampling_params)

# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# 自定义 Dataset 类
class JsonDataset(Dataset):
    def __init__(self, file_path,tokenizer):
        self.data = []
        #print(file_path)
        self.tokenizer = tokenizer
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回 JSON 对象，可以根据需要选择字段
        return self.data[idx]
    
    def custom_collate_fn(self,batch):
        
        instruction = "For an information-seeking dialog, please help reformulate the question into rewrite that can fully express the user's information needs without the need of context. The output should be: 'Rewrite:$rewrite'\n\n"
        
        final_prompt = []
        sample_id = []

        for item in batch:
            sid = item["sample_id"]
            sample_id.append(sid)
            context = item["ctx_utts_text"]
            this_dialog = []
            if not context:
                this_dialog.append("N/A")
            else:
                for i,ctx in enumerate(context): 
                    this_dialog.append("Question: {}".format(ctx))
               
            this_dialog[0] = "Context:\n" + this_dialog[0]
            this_dialog.append("Current Question: " + item["cur_utt_text"])  
            this_dialog = "\n\n".join(this_dialog)
            user_prompt = instruction + this_dialog
            
            
            
            
            messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
            ]
            template_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True)
            
            final_prompt.append(template_prompt)
            


        return {
            'bt_sample_id': sample_id,  
            'bt_prompt': final_prompt 
    }

model_checkpoint_path = "/root/paddlejob/workspace/env_run/rag/LLaMA-Factory/models/llama3_lora_sft"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
sampling_params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=200)
llm = LLM(model=model_checkpoint_path, tensor_parallel_size=8, gpu_memory_utilization=0.3,max_seq_len_to_capture=2048)
