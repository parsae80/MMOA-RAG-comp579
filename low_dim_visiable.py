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
import random

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

from sklearn.decomposition import PCA
import numpy as np

def pca_reduce_to_2d(array):
    """
    使用 PCA 将 n 维 NumPy 数组降为二维。

    :param array: n 维 NumPy 数组，形状为 (samples, features)。
    :return: 二维 NumPy 数组。
    """
    # 初始化 PCA，目标是降到2维
    pca = PCA(n_components=2)
    # 执行降维
    reduced_data = pca.fit_transform(array)
    return reduced_data

import matplotlib.pyplot as plt

def plot_2d_data(data1, data2, labels1='Train', labels2='Test', save_name='./plot_2d.jpg', title='2D Data Visualizatio'):
    """
    可视化两个二维数据集。

    :param data1: 第一个二维数据集（NumPy 数组）。
    :param data2: 第二个二维数据集（NumPy 数组）。
    :param labels1: 第一个数据集的标签。
    :param labels2: 第二个数据集的标签。
    """
    plt.figure(figsize=(8, 6))
    
    # 绘制第一个数据集
    plt.scatter(data1[:, 0], data1[:, 1], c='blue', label=labels1, alpha=0.6)
    
    # 绘制第二个数据集
    plt.scatter(data2[:, 0], data2[:, 1], c='red', label=labels2, alpha=0.6)
    
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_name)

retriever_model_path = '/root/paddlejob/workspace/env_run/rag/models/facebook/contriever'

# loading retriever model
print('*'*20)
print('loading retriever model')
## 检查GPU是否可用  
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_path)
retriever_model = AutoModel.from_pretrained(retriever_model_path)
retriever_model = retriever_model.to(device)
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# loading train set and test set
train_data_path = '/root/paddlejob/workspace/env_run/rag/data/train_top_k_docs.jsonl'
test_data_path = '/root/paddlejob/workspace/env_run/rag/data/val_top_k_docs.jsonl'
train_data, test_data = [], []
with open(train_data_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Strip any leading/trailing whitespace and parse the JSON object
        json_object = json.loads(line)
        train_data.append(json_object)
with open(test_data_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Strip any leading/trailing whitespace and parse the JSON object
        json_object = json.loads(line)
        test_data.append(json_object)

train_questions, test_questions = [], []
for temp_data in train_data:
    train_questions.append(temp_data['answer'])
for temp_data in test_data:
    test_questions.append(temp_data['answer'])
print(len(train_questions), len(test_questions))

# random.shuffle(train_questions)
# train_questions, test_questions = train_questions[:3610], test_questions[:3610]

# get embeddings
batch_size = 1024
train_batch_questions_embeddings, test_batch_questions_embeddings = [], []
for i in tqdm(range(0, len(train_questions), batch_size)):
    batch_questions = train_questions[i: i+batch_size]
    batch_questions_embeddings = get_embeddings(batch_questions)
    train_batch_questions_embeddings.append(batch_questions_embeddings)
all_low_dims = np.concatenate(train_batch_questions_embeddings, axis=0)
print('all_low_dims: {}'.format(all_low_dims.shape))
batch_questions_train_2_dim_array = pca_reduce_to_2d(all_low_dims)
print('batch_questions_train_2_dim_array shape: {}'.format(batch_questions_train_2_dim_array.shape))
print('\n')

for i in tqdm(range(0, len(test_questions), batch_size)):
    batch_questions = test_questions[i: i+batch_size]
    batch_questions_embeddings = get_embeddings(batch_questions)
    test_batch_questions_embeddings.append(batch_questions_embeddings)
all_low_dims = np.concatenate(test_batch_questions_embeddings, axis=0)
print('all_low_dims: {}'.format(all_low_dims.shape))
batch_questions_test_2_dim_array = pca_reduce_to_2d(all_low_dims)
print('batch_questions_test_2_dim_array shape: {}'.format(batch_questions_test_2_dim_array.shape))

plot_2d_data(batch_questions_train_2_dim_array, batch_questions_test_2_dim_array, save_name='./visiable_answer.jpg', title='answers 2d plot')

