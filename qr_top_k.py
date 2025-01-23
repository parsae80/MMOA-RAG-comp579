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

# loading query rewriter data
print('*'*20)
print('loading query rewriter data')
file_path = '/root/paddlejob/workspace/env_run/rag/data/query_rewrite_data/query_rewrite_hotpotqa.json'
with open(file_path, 'r', encoding='utf-8') as file:
    query_rewrite_data = json.load(file)  # 使用 json.load() 将 JSON 数据解析为字典
init_questions = []
rewritten_questions = []
query_rewrite_dict = {}
for i in range(len(query_rewrite_data)):
    init_q = query_rewrite_data[i]['source'].strip()
    rewritten_qs_list = query_rewrite_data[i]['target'].split(';')
    rewritten_qs_list = [q.strip() for q in rewritten_qs_list]
    init_questions.append(init_q)
    rewritten_questions.append(rewritten_qs_list)
    query_rewrite_dict[init_q] = rewritten_qs_list
# get all subqueries
questions = []
for i in range(len(rewritten_questions)):
    for q in rewritten_questions[i]:
        questions.append(q)
print('len(questions): {}'.format(len(questions)))

# questions = questions[:100]
# print(questions)

pre_path = '/root/paddlejob/workspace/env_run/rag/data/'
retriever_model_path = '/root/paddlejob/workspace/env_run/rag/models/facebook/contriever'
index_path = pre_path+'wikipedia.contriever'
docs_path = pre_path+'psgs_w100.tsv'
top_k_docs_path = '/root/paddlejob/workspace/env_run/rag/data/query_rewrite_data/top_k_qr_subq.jsonl'

# loading retriever model
print('*'*20)
print('loading retriever model')
# 检查GPU是否可用  
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_path)
retriever_model = AutoModel.from_pretrained(retriever_model_path)
retriever_model = retriever_model.to(device)
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# 打印可用的 GPU 数量
print("Available GPUs:", torch.cuda.device_count())
# 打印每个 GPU 的名称
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# loading index
print('*'*20)
print('loading index')
start_time = time.time()

# # cpu faiss
# print('using cpu index')
# index = faiss.read_index(index_path)
# gpu faiss
print('load index')
res = faiss.StandardGpuResources()  # 创建一个Faiss资源对象
read_index = faiss.read_index(index_path)  # 读取索引
print('put index on gpu')
index = faiss.index_cpu_to_gpu(res, 0, read_index)  # 把索引放到第一个GPU上

end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# loading docs data
print('*'*20)
print('loading data')
start_time = time.time()
df = pd.read_csv(docs_path, sep='\t')
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# searching
print('*'*20)
print('searching')
start_time = time.time()
top_k_list = []
k=10


batch_size = 10
iterations = len(questions) // batch_size
cur_batch = 0
for b in tqdm(range(iterations)):
    batch_questions = questions[cur_batch*batch_size: min((cur_batch+1)*batch_size, len(questions))]
    batch_questions_embeddings = get_embeddings(batch_questions)
    batch_D, batch_I = index.search(batch_questions_embeddings, k)
    cur_batch += 1
    for q in range(len(batch_questions)):
        temp_dict = {}
        question = batch_questions[q]
        temp_dict['question'] = question
        temp_dict['top_k_docs'] = [{'doc_id': str(doc_id), 'content': str(df.loc[doc_id, 'title'])+'\n' + str(df.loc[doc_id, 'text'])} for doc_id in batch_I[q]]
        top_k_list.append(temp_dict)

# 保存top-k的retrieval结果
with open(top_k_docs_path, 'w') as file:
    for dictionary in top_k_list:
        # json.dumps()函数将字典转化为json字符串
        # 每个字典写入一行
        file.write(json.dumps(dictionary) + '\n')

print('Finished!')
