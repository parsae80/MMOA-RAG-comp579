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


# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]

    return sentence_embeddings

def get_embeddings(sentences):
    # Apply tokenizer
    sentences_input = retriever_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    if torch.cuda.is_available():
        sentences_input = sentences_input.to(device)
        with torch.no_grad():  # 如果不需要梯度，使用torch.no_grad()可以减少内存消耗  
            sentences_output = retriever_model(**sentences_input)

        # Compute token embeddings
        sentences_embedding = mean_pooling(sentences_output[0], sentences_input['attention_mask'])
        
        sentences_embedding = sentences_embedding.cpu().numpy()

        # Delete variables and empty cache
        del sentences_input, sentences_output
        torch.cuda.empty_cache()

    else:
        sentences_output = retriever_model(**sentences_input) 
        sentences_embedding = mean_pooling(sentences_output[0], sentences_input['attention_mask'])
        sentences_embedding = sentences_embedding.numpy()

    return sentences_embedding

def normalize_answer_final(answer):
    pre_answer = answer.split('\n\n')[-1].split('Answer: ')[-1].split('The answer is: ')[-1]
    final_answer = normalize_answer(pre_answer)
    return final_answer

def extract_digits(input_string, K_candidate):
    input_list = input_string.split(',')
    input_list = [item.replace("Document", "") for item in input_list]

    # 创建一个空列表来存储结果
    digits_list = []
    
    # 遍历输入字符串中的每个字符
    candidate_list = [str(i) for i in range(K_candidate)]
    for char in input_list:
        # 检查字符是否为数字
        if char in candidate_list:
            # 将字符转换为整数并添加到列表中
            digits_list.append(int(char))

    # K_candidate之后的不要
    digits_list = digits_list[: K_candidate]

    # 去重复
    my_list = digits_list
    unique_list = []
    for item in my_list:
        if item not in unique_list:
            unique_list.append(item)

    # 消除不在范围内的
    return_list = []
    for item in unique_list:
        if item >= 0 and item < K_candidate:
            return_list.append(item)
    
    return return_list

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
            max_new_tokens=1,
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
    input_content = "Question is: {}\n\n".format(str(question))
    for doc_id in range(len(top_k_docs)):
        doc_content = top_k_docs[doc_id]#['content']
        input_content = input_content + "Document {}: {}\n\n".format(str(doc_id), str(doc_content))

    message = [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Your task is to output the ID of the candidate Documents (0,1,2,...,{}) which are helpful in answering the Question.".format(len(top_k_docs)-1)}, 
        {'role': 'assistant', 'content': 'Okay, I will provide the ID of candidate Documents which are helpful in answering the Question.'},
        {'role': 'user', 'content': input_content},
        {'role': 'assistant', 'content': "OK, I received the Question and the candidate Documents."}
    ]

    return message

def get_generator_prefix_role_prompt(question, top_k_docs):
    input_content = "Question is: {}\n\n".format(str(question))
    for doc_id in range(len(top_k_docs)):
        doc_content = top_k_docs[doc_id]#['content']
        input_content = input_content + "Document {}: {}\n\n".format(str(doc_id), str(doc_content))

    if len(top_k_docs) > 0:
        input_content = input_content + "Now, answer the Question: {}, based on the above Documents".format(str(question))
        message = [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question based on the given documents. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."},
        {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question based on the corresponding documents. Please provide the question and the corresponding documents.'},
        {'role': 'user', 'content': input_content},
        {'role': 'assistant', 'content': "OK, I received the Question and the corresponding Documents."}
    ]

    elif len(top_k_docs) == 0:
        input_content = input_content + "Now, answer the Question: {}.".format(str(question))
        message = [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."},
        {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question. Please provide the question.'},
        {'role': 'user', 'content': input_content},
        {'role': 'assistant', 'content': "OK, I received the Question."}
    ]

    return message

def get_selector_post_role_prompt(question, top_k_docs):

    return {'role': 'user', 'content': "Now, output the ID of the candidate Documents (0,1,2,...,{}) which are helpful in answering the Question: {}, for example, in the following format: Document0,Document4,Document6,Document7.".format(len(top_k_docs)-1, str(question), len(top_k_docs)-1)}

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

def get_qr_messages(question):
    messages = [
        {'role': 'system', 'content': "You are a professional assistant skilled at rewriting complex or unclear questions into simpler, more searchable subquestions."},
        {'role': 'assistant', 'content': 'Okay, I will provide the rewritten sub-questions.'},
        {'role': 'user', 'content': "Please help me rewrite or decompose the given questions into sub-questions, making them easier to search for answers in a search engine. The rewritten sub-questions must have logical connections and dependencies, without being overly repetitive in meaning. Additionally, avoid using vague demonstrative pronouns and similar terms"},
        {'role': 'assistant', 'content': 'Okay, I will provide the rewritten sub-questions.'},
        {'role': 'user', 'content': "Original question is '{}'. Now rewrite or decompose the original question into sub-questions according to the above requirements, and only output the rewritten subquestions in the format of one subproblem per line without any additional content. Additionally, avoid using vague demonstrative pronouns and similar terms, avoid the duplicate subquestions.".format(question)}
    ]

    return messages

def get_yes_or_no_messages(question):
    messages = [
        {'role': 'system', 'content': "You are a professional assistant who is good at judging whether you can directly answer the questions given."},
        {'role': 'assistant', 'content': 'Okay, please provide your demand.'},
        {'role': 'user', 'content': "Would you please judge whether the question given can be answered directly? If you are sure your answer is correct, answer yes. If you're not sure, answer no, and we'll provide more relevant documents on the question later."},
        {'role': 'assistant', 'content': 'Okay, please provide the question.'},
        {'role': 'user', 'content': "The question is '{}'. Please answer: yes or no, as requested".format(question)}
    ]

    return messages

def get_init_messages(question, doc):
    messages = [
        {'role': 'system', 'content': "You are a professional assistant who is good at finding pieces or evidence in a given document that are helpful in answering the given question."},
        {'role': 'assistant', 'content': 'Okay, please provide your question and document.'},
        {'role': 'user', 'content': "Find pieces of evidence in a given document that are helpful in answering a given question."},
        {'role': 'assistant', 'content': 'Okay, please provide the question and document.'},
        {'role': 'user', 'content': "The question is '{}'\nThe document is '{}'\nOnly output the pieces of evidence in a given document that are helpful in answering a given question and nothing else explanation.".format(question, doc)}
    ]

    return messages

def get_final_messages(question, top_k_docs, support_passages):
    input_content = "Question is: {}\n\n".format(str(question))
    for doc_id in range(len(top_k_docs)):
        doc_content = top_k_docs[doc_id]#['content']
        support_evidence = support_passages[doc_id]
        input_content = input_content + "Document {}: {}\n" + "Support evidence of Document {}: {}\n" + "\n\n".format(str(doc_id), str(doc_content), str(doc_id), str(support_evidence))

    input_content = input_content + "Now, answer the Question: {}, based on the above Documents".format(str(question))
    messages = [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question based on the given documents. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."},
        {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question based on the corresponding documents. Please provide the question and the corresponding documents.'},
        {'role': 'user', 'content': input_content},
        {'role': 'assistant', 'content': "OK, I received the Question and the corresponding Documents."},
        {'role': 'user', 'content': "Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer with the form of **answer** and nothing else."}
    ]

    return messages


dataset_name = '2wikimultihopqa'

top_k_docs_path = '/root/paddlejob/workspace/env_run/rag/data/{}/val_top_k_docs.jsonl'.format(dataset_name)
# generator_model_path = '/root/paddlejob/workspace/env_run/rag/LLaMA-Factory/models/llama3-8b/{}/qr_selector_and_generator/ppo_1'.format(dataset_name)
yes_or_no_model_path = '/root/paddlejob/workspace/env_run/rag/models/LLM-Research/Meta-Llama-3-8B-Instruct'
sft_model_path = '/root/paddlejob/workspace/env_run/rag/LLaMA-Factory/models/llama3-8b/{}/qr_selector_and_generator/sft_1'.format(dataset_name)
init_model_path = '/root/paddlejob/workspace/env_run/rag/models/LLM-Research/Meta-Llama-3-8B-Instruct'

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
yes_or_no_model = AutoModelForCausalLM.from_pretrained(
    yes_or_no_model_path,
    torch_dtype="auto",
    device_map="auto"
)
yes_or_no_model.eval()
yes_or_no_model = yes_or_no_model.to(torch.float16)
yes_or_no_tokenizer = AutoTokenizer.from_pretrained(yes_or_no_model_path)
yes_or_no_tokenizer.padding_side = "left"
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# loading generator model
print('*'*20)
print('loading generator model')
sft_model = AutoModelForCausalLM.from_pretrained(
    sft_model_path,
    torch_dtype="auto",
    device_map="auto"
)
sft_model.eval()
sft_model = sft_model.to(torch.float16)
sft_tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
sft_tokenizer.padding_side = "left"
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# loading generator model
print('*'*20)
print('loading generator model')
init_model = AutoModelForCausalLM.from_pretrained(
    init_model_path,
    torch_dtype="auto",
    device_map="auto"
)
init_model.eval()
init_model = init_model.to(torch.float16)
init_tokenizer = AutoTokenizer.from_pretrained(init_model_path)
init_tokenizer.padding_side = "left"
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# top_k_list = top_k_list[:100]

all_questions = []
all_answers = []
all_top_docs = []
shuffled_all_top_docs = []
for i in range(len(top_k_list)):
    cur_que_ans_docs = top_k_list[i]
    all_questions.append(cur_que_ans_docs['question'])
    all_answers.append(cur_que_ans_docs['answer'])
    all_top_docs.append([doc['content'] for doc in cur_que_ans_docs['top_k_docs']])
    cur_top_docs_content_list = [doc_item['content'] for doc_item in cur_que_ans_docs['top_k_docs']]
    shuffled_all_top_docs.append(cur_top_docs_content_list)

# 定义允许的token
allowed_tokens=['yes', 'no']
allowed_token_ids = yes_or_no_tokenizer.convert_tokens_to_ids(allowed_tokens)
eos_token_id = yes_or_no_tokenizer.eos_token_id
allowed_token_ids.append(eos_token_id)
# 创建LogitsProcessor
logits_processor = LogitsProcessorList([
    AllowedTokensLogitsProcessor(allowed_token_ids)
])

# *******************************SELF-RAG*******************************

# answers
generator_answers = []

for i in tqdm(range(len(all_questions))):
    question = all_questions[i]
    top_docs = shuffled_all_top_docs[i]

    # yes or no: retriever
    message_yes_or_no = [get_yes_or_no_messages(question=question)]
    answer_yes_or_no = batch_get_response(yes_or_no_model, yes_or_no_tokenizer, message_yes_or_no, logits_processor)

    if answer_yes_or_no[0] == 'no':
        # get final answer
        message_sft = [get_messages(question=question, top_docs=top_docs)]
        answer_sft = batch_get_response(sft_model, sft_tokenizer, message_sft)
        answer_sft = answer_sft[0].replace("*", "")
        generator_answers.append(answer_sft)

        if i < 10:
            print(answer_sft)

    elif answer_yes_or_no[0] == 'yes':
        # get init messages
        messages_init = []
        for doc in top_docs:
            message_doc = get_init_messages(question=question, doc=doc)
            messages_init.append(message_doc)
        answers_init = batch_get_response(init_model, init_tokenizer, messages_init)
        support_passages_list = []
        for answer_init in answers_init:
            answer_init = answer_init.replace("*", "")
            support_passages_list.append(answer_init)
        # get final answer
        message_final = [get_final_messages(question=question, top_k_docs=top_docs, support_passages=support_passages_list)]
        answer_final = batch_get_response(sft_model, sft_tokenizer, message_final)
        answer_final = answer_final[0].replace("*", "")
        generator_answers.append(answer_final)

        if i < 10:
            print(answer_final)

# getting F1 score
print(compute_scores(generator_answers, all_answers))

save_results_path = "/root/paddlejob/workspace/env_run/rag/data/{}".format(dataset_name)
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)
# 保存答案为jsonl
with open(save_results_path+'/SELF-RAG.jsonl', 'w') as file: # sft_2_g.jsonl
    for i in range(len(generator_answers)):
        temp_dic = {'question_id': str(i), 'question': str(all_questions[i]), 'predict_answer': str(generator_answers[i]), 'golden_answer': str(all_answers[i])}
        file.write(json.dumps(temp_dic) + '\n')

end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

print('Finished!')