# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/ppo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import sys
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits
from trl.models.utils import unwrap_model_for_generation
from typing_extensions import override

from ...extras.logging import get_logger
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .ppo_utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm

import re
import time
import json
import random
import faiss
import requests
import pandas as pd
from .normalize_answers import *
from collections import Counter
from fastchat.model import get_conversation_template
from transformers import LogitsProcessorList, LogitsProcessor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)

def remove_punctuation(text):
    # 定义要去除的标点符号
    punctuation = set('.,!?;:"()[]{}-')
    # 过滤掉标点符号
    return ''.join(char for char in text if char not in punctuation)

def clean_and_split(text):
    # 去除标点符号
    cleaned_text = remove_punctuation(text).lower()
    # 以空格为分隔符分割单词
    words = cleaned_text.split()
    return words

def calculate_match_ratio(answer, document):
    # 常见词汇列表，包括介词、冠词、代词、连词、助动词、否定词和疑问词
    common_words = {
        # 常见介词、冠词、代词、连词、助动词、否定词、疑问词
        "in", "on", "at", "to", "for", "with", "by", "from", "about",
        "a", "an", "the",
        "it", "they", "we", "you", "he", "she", "i", "me", "my", "mine", "ours", "us", "your", "yours", "his", "hers", "their", "theirs",
        "and", "or", "but", "because", "if", "then", "than", "as",
        "is", "are", "was", "were", "do", "does", "did", "have", "has", "had", "having", "be", "been", "being",
        "not", "no", "nor", "none",
        "what", "where", "when", "who", "why", "how", "which", "whom", "whose",
        # 常用标点符号
        ".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", "{", "}", "\"", "'", "...", "--", "/", "\\", "|", "<", ">", "=", "+", "*", "&", "^", "%", "$", "#", "@", "~", "`",
        # 其他常见停用词
        "of", "that", "this", "these", "those", "such", "there", "here", "all", "any", "both", "each", "few", "more", "some", "most", "other", "another", "every", "either", "neither"
    }
    # 清理并分割answer和document
    answer_words = [word for word in clean_and_split(answer) if word not in common_words]
    document_words = remove_punctuation(document).lower()
    # 计算answer中有多少单词出现在document中
    match_count = sum(1 for word in answer_words if word in document_words)
    # 计算比例
    if len(answer_words) == 0:
        return 0.0  # 避免除以0
    match_ratio = match_count / (2*len(answer_words))

    return match_ratio

def sort_and_classify_documents(answer, documents):
    # 计算每个文档的匹配比例
    document_ratios = [(document, calculate_match_ratio(answer, document)) for document in documents]

    return_binary_list = [0] * len(document_ratios)
    for i in range(len(document_ratios)):
        doc_ratio = document_ratios[i]
        if doc_ratio[1] > 0:
            return_binary_list[i] = 1
        elif doc_ratio[1] == 0:
            pass

    return return_binary_list

def get_selector_metrics(predict_answer, golden_answer, candidate_documents):
    predict_binary_list = sort_and_classify_documents(predict_answer, candidate_documents)
    golden_binary_list = sort_and_classify_documents(golden_answer, candidate_documents)

    # 计算各项指标
    accuracy = accuracy_score(golden_binary_list, predict_binary_list)
    precision = precision_score(golden_binary_list, predict_binary_list)
    recall = recall_score(golden_binary_list, predict_binary_list)
    f1 = f1_score(golden_binary_list, predict_binary_list)

    # # 打印结果
    # print('predict_binary_list:', predict_binary_list, '  golden_binary_list:', golden_binary_list)
    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1 Score: {f1}")
    # print('\n')

    return f1

# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]

    return sentence_embeddings

def get_embeddings(sentences, retriever_model, retriever_tokenizer):
    # Apply tokenizer
    sentences_input = retriever_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    if torch.cuda.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        sentences_input = sentences_input.to(device)  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
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

def extract_questions_and_docs(response):
    """
    从HTTP响应中提取问题和对应的文档列表。

    参数：
    - response: requests.Response 对象，HTTP请求的响应。

    返回：
    - 提取的信息列表，每个元素是一个字典，包含key 'question' 和value 'top_k_docs'。
    """

    # 解析 JSON 响应
    results = response.json()

    # 提取信息
    extracted_info_dict = {}
    for result in results:
        question = result.get('question')
        top_k_docs = result.get('top_k_docs', [])
        extracted_info_dict[question] = top_k_docs

    return extracted_info_dict


class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)
    
    def __call__(self, input_ids, scores):
        # 获取scores张量所在的设备
        device = scores.device
        # 在相同设备上创建mask张量
        mask = torch.full(scores.shape, float('-inf'), device=device)
        mask[:, list(self.allowed_token_ids)] = 0
        # 确保scores和mask在同一设备上进行计算
        scores = scores + mask
        
        return scores


class CustomPPOTrainer_Ablation_S_G(PPOTrainer, Trainer):
    r"""
    Inherits PPOTrainer.
    """

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[List["TrainerCallback"]],
        model: "AutoModelForCausalLMWithValueHead",
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_collator: "DataCollatorWithPadding",
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
    ) -> None:
        if eval_dataset is not None:
            raise NotImplementedError("PPOTrainer does not support eval dataset yet.")

        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
        )

        # Add deepspeed config
        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with is not None:
                logger.warning("PPOTrainer cannot use external logger when DeepSpeed is enabled.")
                ppo_config.log_with = None

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(train_dataset) / total_train_batch_size
            )

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        # self.reward_model = reward_model
        self.current_device = get_current_device()  # patch for deepspeed training

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
        )
        if self.args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        self.amp_context = torch.autocast(self.current_device.type)
        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        # if finetuning_args.reward_model_type == "full":
        #     if self.is_deepspeed_enabled:
        #         if not (
        #             getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
        #             or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
        #         ):  # quantized models are already set on the correct device
        #             self.reward_model = self._prepare_deepspeed(self.reward_model)
        #     else:
        #         self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # 更新n个模块 batch_size需要扩大n倍
        self.batch_size_1 = self.config.batch_size
        self.batch_size_2 = 2 * self.batch_size_1
        self.batch_size_3 = 3 * self.batch_size_1


    def extract_question(self, text):
        """
        从给定的文本中提取问题部分。
        
        参数:
        text (str): 包含问题和其他内容的字符串。
        
        返回:
        str: 提取出的问题字符串，如果没有找到则返回空字符串。
        """
        # 使用非贪婪模式 .*? 来匹配 "Question is:" 和 "Document 0" 之间的内容
        match = re.search(r'Question is:(.*?)Document0', text, re.DOTALL)
        if match:
            return match.group(1).strip()  # 使用 strip() 去掉首尾可能的空格
        else:
            print("warning: cannot find the question.")
            return ""

    def extract_document(self, text, doc_number):
        """
        提取指定编号的 Document 的内容。
        
        参数:
        text (str): 包含多个 Document 的字符串。
        doc_number (int): 要提取的 Document 编号。
        
        返回:
        str: 提取出的 Document 内容，如果没有找到则返回空字符串。
        """
        if doc_number < 9:
            # 对于 Document 0 到 8，使用常规的模式匹配
            pattern = rf'Document{doc_number}:(.*?)(?=Document{doc_number + 1}:|$)'
        else:
            # 对于 Document 9，使用 \n\nNow 作为结束标记
            pattern = rf'Document{doc_number}:(.*?)(?=\n\nNow)'
        
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()  # 去掉首尾可能的空格
        else:
            print(f"warning: cannot find Document {doc_number}.")
            return ""

    def convert_to_int_list(self, input_string):
        # 创建一个空列表来存储结果
        digits_list = []
        
        # 遍历输入字符串中的每个字符
        for char in input_string:
            # 检查字符是否为数字
            if char.isdigit():
                # 将字符转换为整数并添加到列表中
                digits_list.append(int(char))
        
        return digits_list

    def extract_digits(self, input_string, K_candidate):
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

    def get_selector_duplicate_reward(self, input_string, K_candidate):
        duplicate_reward = 0.0

        # 如果回答为空 直接返回0.0
        if input_string == '':
            return duplicate_reward

        # 定义正则表达式模式，匹配以"Document"开头，后面跟一个或多个数字的格式
        pattern = r"^(Document\d+,)*(Document\d+)$"
        
        # 检查输入字符串是否匹配模式
        if not re.match(pattern, input_string):
            duplicate_reward += -0.5
        
        # 将字符串分割成数字列表
        input_list = input_string.split(',')
        input_list = [item.replace("Document", "") for item in input_list]
        numbers = input_list

        # 检查是否有非法数字
        candidate_list = [str(i) for i in range(K_candidate)]
        for number in numbers:
            if number not in candidate_list:
                duplicate_reward += -0.5
        # 检查是否有重复数字
        if len(numbers) != len(set(numbers)):
            duplicate_reward += -0.5
        
        return -1.0 if duplicate_reward < 0 else 0.0

    def get_answer_dict(self, answers_path):
        
        # loading answers file
        print('loading pairwise data to get golden_answer dict.')
        start_time = time.time()
        # 创建一个空列表来存放字典
        answers_pair = []
        # 打开文件以读取
        with open(answers_path, 'r') as file:
            for line in file:
                # json.loads() 函数将每行的json字符串转化为字典
                answers_pair.append(json.loads(line))
        end_time = time.time()
        print('time consuming: {} seconds'.format(end_time - start_time))

        question_golden_answer_dict = {}
        for ans_pair in answers_pair:
            # question_golden_answer_dict[ans_pair['question'].replace(" ", "")] = ans_pair['golden_answer']  # nq_open
            question_golden_answer_dict[ans_pair['question'].replace(" ", "")] = ans_pair['answer']  # hotpotqa 2wikimultihopqa

        return question_golden_answer_dict

    def normalize_answer_final(self, answer):
        pre_answer = answer.split('\n\n')[-1].split('Answer: ')[-1].split('The answer is: ')[-1]
        final_answer = normalize_answer(pre_answer)
        return final_answer

    def get_rewards(self, predict_answers, golden_answers, reward_metric_name='f1'):  # torch.rand((self.config.mini_batch_size, 1))

        assert len(predict_answers) == len(golden_answers), "预测答案和标准答案的长度不相等"

        rewards = []
        for i in range(len(predict_answers)):
            # reward: metrics
            normalized_prediction = self.normalize_answer_final(predict_answers[i])
            normalized_ground_truth = self.normalize_answer_final(golden_answers[i])
            reward_metric = {"acc": 0.0, "em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

            if normalized_prediction == normalized_ground_truth:
                reward_metric['em'] = 1.0

            if normalized_ground_truth in normalized_prediction:# or normalized_prediction in normalized_ground_truth:
                reward_metric["acc"] = 1.0

            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            
            if len(prediction_tokens) != 0:
                precision = 1.0 * num_same / len(prediction_tokens)
            else:
                precision = 0.0

            if len(ground_truth_tokens) != 0:
                recall = 1.0 * num_same / len(ground_truth_tokens)
            else:
                recall = 0.0

            if precision + recall != 0:
                f1 = (2 * precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            reward_metric['f1'], reward_metric['precison'], reward_metric['recall'] = f1, precision, recall
            
            rewards.append(reward_metric[reward_metric_name])
            # temp_reward = (reward_metric['f1']+reward_metric['em']) / 2
            # rewards.append(temp_reward)

        return torch.tensor(rewards).view(-1, 1)

    def get_generator_punish(self, predict_answers):
        rewards = []
        for i in range(len(predict_answers)):
            predict_answer = predict_answers[i]
            words = predict_answer.split()
            word_count = len(words)

            score = 0.0
            if word_count > 30:
                score += -1.0
            
            rewards.append(score)
        
        return torch.tensor(rewards).view(-1, 1)

    def get_selector_rewards(self, predict_answers, golden_answers, mini_batch_candidate_docs, mini_batch_input_questions, mini_batch_selector_answers_text):

        assert len(predict_answers) == len(golden_answers), "预测答案和标准答案的长度不相等"

        rewards = []
        for i in range(len(predict_answers)):
            K_candidate_num = len(mini_batch_candidate_docs[i])
            score = self.get_selector_duplicate_reward(mini_batch_selector_answers_text[i], K_candidate_num)  # 0.0 or -0.5 (惩罚输出重复内容和非doc id的内容)
            if golden_answers[i] == 'yes' or golden_answers[i] == 'no':
                score += get_selector_metrics(predict_answers[i], mini_batch_input_questions[i], mini_batch_candidate_docs[i]) # yes or no的问题
            else:
                score += get_selector_metrics(predict_answers[i], golden_answers[i], mini_batch_candidate_docs[i]) # 非 yes or no

            rewards.append(score)

        return torch.tensor(rewards).view(-1, 1)

    def get_selector_repeat_punish(self, mini_batch_selector_answers_text, mini_batch_candidate_docs):
        rewards = []
        for i in range(len(mini_batch_selector_answers_text)):
            K_candidate_num = len(mini_batch_candidate_docs[i])
            score = self.get_selector_duplicate_reward(mini_batch_selector_answers_text[i], K_candidate_num)  # 0.0 or -0.5 (惩罚输出重复内容和非doc id的内容)
            rewards.append(score)
        
        return torch.tensor(rewards).view(-1, 1)

    def get_qr_punish(self, predict_answers, mini_batch_subquestions):
        rewards = []
        for i in range(len(predict_answers)):
            if len(mini_batch_subquestions[i]) > 4:  # subq个数惩罚
                score = -0.5
            else:
                score = 0.0
            
            sum_char_num = 0
            for subq in mini_batch_subquestions[i]:
                sum_char_num += len(subq)
            if sum_char_num > 300:
                score = -0.5
                
            rewards.append(score)
        
        return torch.tensor(rewards).view(-1, 1)

    def get_selector_prefix_role_prompt(self, question, top_k_docs):
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

    def get_generator_prefix_role_prompt(self, question, top_k_docs):
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

    def get_selector_post_role_prompt(self, question, top_k_docs):

        return {'role': 'user', 'content': "Now, output the ID of the candidate Documents (0,1,2,...,{}) which are helpful in answering the Question: {}, for example, in the following format: Document0,Document4,Document6,Document7.".format(len(top_k_docs)-1, str(question), len(top_k_docs)-1)}

    def get_generator_post_role_prompt(self, top_docs):
        if len(top_docs) > 0:
            message = {'role': 'user', 'content': "Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer with the form of **answer** and nothing else."}
        elif len(top_docs) == 0:
            message = {'role': 'user', 'content': "Given the Question, predict the answer to the Question as briefly and accurately as possible. Only give the brief and accurate answer with the form of **answer** and nothing else."}

        return message

    def get_generator_messages(self, question, top_docs):
        messages = self.get_generator_prefix_role_prompt(question, top_docs)
        messages.append(self.get_generator_post_role_prompt(top_docs))

        return messages

    def get_selector_messages(self, question, top_docs):
        messages = self.get_selector_prefix_role_prompt(question, top_docs)
        messages.append(self.get_selector_post_role_prompt(question, top_docs))

        return messages

    def get_qr_messages(self, question):
        messages = [
            {'role': 'system', 'content': "You are a professional assistant skilled at rewriting complex or unclear questions into simpler, more searchable subquestions."},
            {'role': 'assistant', 'content': 'Okay, I will provide the rewritten sub-questions.'},
            {'role': 'user', 'content': "Please help me rewrite or decompose the given questions into sub-questions, making them easier to search for answers in a search engine. The rewritten sub-questions must have logical connections and dependencies, without being overly repetitive in meaning. Additionally, avoid using vague demonstrative pronouns and similar terms"},
            {'role': 'assistant', 'content': 'Okay, I will provide the rewritten sub-questions.'},
            {'role': 'user', 'content': "Original question is '{}'. Now rewrite or decompose the original question into sub-questions according to the above requirements, and only output the rewritten subquestions in the format of one subproblem per line without any additional content. Additionally, avoid using vague demonstrative pronouns and similar terms, avoid the duplicate subquestions.".format(question)}
        ]

        return messages

    def trans_text_to_token(self, messages_list):
        # Create a list of input_ids for all messages in the batch
        input_ids_list = []
        for messages in messages_list:
            if self.tokenizer.chat_template is not None:
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ).cuda()
            else:
                conv = get_conversation_template(generator_model_path)
                for message in messages:
                    conv.append_message(message["role"], message["content"])
                conv.append_message("assistant", "")
                input_ids = self.tokenizer(conv.get_prompt(), return_tensors="pt").input_ids.cuda()
            
            input_ids_list.append(input_ids)

        # Pad input_ids to the same length and create a single tensor
        input_ids_list = [input_ids.squeeze(0) for input_ids in input_ids_list]  # 向量需要是一维的

        # print('messages_list', messages_list)
        # print('input_ids_list', input_ids_list)

        # 找出最长序列的长度
        max_length = max(input_ids.size(0) for input_ids in input_ids_list)
        # 手动左填充
        input_ids_padded = torch.stack([
            torch.cat([input_ids.new_full((max_length - input_ids.size(0),), self.tokenizer.eos_token_id), input_ids], dim=0)
            for input_ids in input_ids_list
        ], dim=0)

        # 创建 attention mask
        attention_masks = torch.stack([
            torch.cat([torch.zeros(max_length - input_ids.size(0), dtype=torch.long), torch.ones(input_ids.size(0), dtype=torch.long)], dim=0)
            for input_ids in input_ids_list
        ], dim=0).cuda()

        temp_batch = {}
        temp_batch["input_ids"] = input_ids_padded
        temp_batch["attention_mask"] = attention_masks

        return temp_batch

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""
        Implements training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.
        """
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info("  Num examples = {:,}".format(num_examples))
            logger.info("  Num Epochs = {:,}".format(num_train_epochs))
            logger.info("  Instantaneous batch size per device = {:,}".format(self.args.per_device_train_batch_size))
            logger.info(
                "  Total train batch size (w. parallel, buffer, distributed & accumulation) = {:,}".format(
                    total_train_batch_size
                )
            )
            logger.info("  Gradient Accumulation steps = {:,}".format(self.args.gradient_accumulation_steps))
            logger.info("  Num optimization epochs per batch = {:,}".format(self.finetuning_args.ppo_epochs))
            logger.info("  Total training steps = {:,}".format(max_steps))
            logger.info("  Number of trainable parameters = {:,}".format(count_parameters(self.model)[0]))

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # 构造{question: golden_answer}的dict
        answers_path = '/root/paddlejob/workspace/env_run/rag/data/2wikimultihopqa/train_top_k_docs.jsonl'
        questions_golden_answers_dict = self.get_answer_dict(answers_path)

        # selector: 定义允许的token
        # allowed_tokens=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', ',', 'Document']
        allowed_tokens=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', 'Document']
        allowed_token_ids = self.tokenizer.convert_tokens_to_ids(allowed_tokens)
        eos_token_id = self.tokenizer.eos_token_id
        allowed_token_ids.append(eos_token_id)
        # selector: 创建LogitsProcessor
        logits_processor = LogitsProcessorList([
            AllowedTokensLogitsProcessor(allowed_token_ids)
        ])

        # 打开文件进行追加写入
        kl_ctl_results_path = self.args.output_dir + '/kl_ctl.txt'
        with open(kl_ctl_results_path, 'a') as file:
            # 将列表转换为字符串并用空格连接，写入一行
            file.write('self.config.init_kl_coef: {}, self.config.target: {}, self.config.horizon: {}'.format(self.config.init_kl_coef, self.config.target, self.config.horizon) + '\n\n\n')

        print('self.batch_size_1: {}, self.config.mini_batch_size: {}'.format(self.batch_size_1, self.config.mini_batch_size))

        self.config.ppo_epochs = 2
        print('self.config.ppo_epochs: {}'.format(self.config.ppo_epochs))

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):

            # if step < 10:
            #     self.config.ppo_epochs = 4
            # else:
            #     self.config.ppo_epochs = 2

            # 打开文件进行追加写入
            kl_ctl_results_path = self.args.output_dir + '/kl_ctl.txt'
            with open(kl_ctl_results_path, 'a') as file:
                # 将列表转换为字符串并用空格连接，写入一行
                file.write('step: {}, '.format(step) + 'self.kl_ctl.value: {}'.format(self.kl_ctl.value) + '\n')

            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            # Get inputs
            self.model.eval()
            self.tokenizer.padding_side = "right"  # change padding side
            queries, responses, rewards = [], [], []

            QR_STEP = 10000000
            SELECTOR_STEP = 5
            GENERATOR_STEP= 1
            CRITIC_STEP = -1
            
            if step < CRITIC_STEP:
                self.config.batch_size = self.batch_size_3  # 同时优化3个模块
            else:
                num = 0
                if step % QR_STEP == 0 and step > 0:
                    num += 1
                if step % SELECTOR_STEP == 0:
                    num += 1
                if step % GENERATOR_STEP == 0:
                    num += 1
                self.config.batch_size = num * self.batch_size_1  # 同时优化num个模块

            torch.cuda.empty_cache()  # 清理optimization过程未使用的显存

            rollout_start_time = time.time()

            # for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
            for idx in range(0, self.batch_size_1, self.config.mini_batch_size):

                # get init text from mini_batch
                mini_batch = batch[idx : idx + self.config.mini_batch_size]

                init_texts = []
                for sub_idx in range(self.config.mini_batch_size):
                    init_text = self.tokenizer.decode(mini_batch["input_ids"][sub_idx], skip_special_tokens=True)
                    init_texts.append(init_text)
                
                mini_batch_input_questions = []
                mini_batch_input_candidate_docs = []
                for text in init_texts:
                    question = self.extract_question(text)
                    mini_batch_input_questions.append(question)
                    temp_docs = []
                    K_init_candidate = 10
                    for k in range(K_init_candidate):
                        temp_doc = self.extract_document(text, k)
                        temp_docs.append(temp_doc)
                    # random.shuffle(temp_docs)
                    mini_batch_input_candidate_docs.append(temp_docs)

                # ****************************************** query rewriter ******************************************
                mini_batch_qr_messages_list = []
                for i in range(len(mini_batch_input_questions)):
                    init_question = mini_batch_input_questions[i]
                    qr_messages = self.get_qr_messages(init_question)
                    mini_batch_qr_messages_list.append(qr_messages)
                
                ## encode text to input_ids
                mini_batch_token_with_mask = self.trans_text_to_token(mini_batch_qr_messages_list)
                ## get selected IDs of docs
                mini_batch_qr_inputs, mini_batch_qr_answers = self.get_inputs(mini_batch_token_with_mask)
                mini_batch_qr_answers_text = []
                for tem_i in range(len(mini_batch_qr_answers)):
                    temp_answer = self.tokenizer.decode(mini_batch_qr_answers[tem_i], skip_special_tokens=True)
                    mini_batch_qr_answers_text.append(temp_answer)
                mini_batch_subquestions = []
                for q_id in range(len(mini_batch_qr_answers_text)):
                    subq_text = mini_batch_qr_answers_text[q_id]
                    qr_answer_list = subq_text.split('\n')
                    qr_answer_list = [q.strip() for q in qr_answer_list]

                    if len(qr_answer_list) > 4 and len(qr_answer_list) <= 8:  # 5 6 7 8
                        qr_answer_list = qr_answer_list[:4]  # 改写后的subquestion不超过4个
                        print('改写subquestion超过了4 (5 6 7 8)个，只取前四个。')
                    elif len(qr_answer_list) > 8:
                        qr_answer_list = [mini_batch_input_questions[q_id]]
                        print('改写subquestion超过了8个，存在问题，替换为原始question。')

                    mini_batch_subquestions.append(qr_answer_list)
                    init_q = mini_batch_input_questions[q_id]

                    # 打开文件进行追加写入
                    qr_results_path = self.args.output_dir + '/context_query_rewriting.txt'
                    with open(qr_results_path, 'a') as file:
                        # 将列表转换为字符串并用空格连接，写入一行
                        file.write(init_q + '\n')
                        for subq in qr_answer_list:
                            file.write(subq + '\n')
                        file.write('\n')

                if step < CRITIC_STEP:
                    queries.extend(mini_batch_qr_inputs)
                    responses.extend(mini_batch_qr_answers)
                else:
                    if step % QR_STEP == 0 and step > 0:
                        queries.extend(mini_batch_qr_inputs)
                        responses.extend(mini_batch_qr_answers)

                # ************** get retrieval docs **************
                questions = []
                for subq_list in mini_batch_subquestions:
                    for subq in subq_list:
                        questions.append(subq)
                
                current_device = torch.cuda.current_device()  # 获取当前正在使用的 GPU 编号
                # print(current_device)
                url = 'http://10.215.192.149:800{}/search'.format(current_device)
                data = {
                    'questions': questions,
                    'N': 10
                }
                response = requests.post(url, json=data)
                qr_all_subquestions_docs_dict = extract_questions_and_docs(response)

                shuffled_all_top_docs = []
                for i in range(len(mini_batch_subquestions)):
                    subq_list = mini_batch_subquestions[i][:4]  # 改写后的query不超过4个
                    num_docs_per_subq_dict = {'1': [10], '2': [5,5], '3': [4,3,3], '4': [3,3,2,2], '5': [2,2,2,2,2]}  # 分别为10 10 12 12 10个候选文档
                    num_docs_per_subq = num_docs_per_subq_dict[str(len(subq_list))]
                    # temp_doc_list = all_top_docs[:4]  # 原问题的前4 doc + subquestion的doc
                    temp_doc_list = []  # 只有subquestion的doc
                    # for subq in subq_list:
                    for subq_id in range(len(subq_list)):
                        subq = subq_list[subq_id]
                        temp_doc_list = temp_doc_list + qr_all_subquestions_docs_dict[subq][: num_docs_per_subq[subq_id]]
                    shuffled_all_top_docs.append(temp_doc_list)
                
                mini_batch_input_candidate_docs = shuffled_all_top_docs

                # ****************************************** selector ******************************************
                ## construct input messages of selector
                mini_batch_messages_list = []
                for batch_i in range(len(mini_batch_input_questions)):
                    question = mini_batch_input_questions[batch_i]
                    top_docs = mini_batch_input_candidate_docs[batch_i]
                    selector_messages = self.get_selector_messages(question, top_docs)
                    mini_batch_messages_list.append(selector_messages)

                    # 打开文件进行追加写入
                    question_results_path = self.args.output_dir + '/context_question.txt'
                    with open(question_results_path, 'a') as file:
                        # 将列表转换为字符串并用空格连接，写入一行
                        file.write(question + '\n')

                ## encode text to input_ids
                mini_batch_token_with_mask = self.trans_text_to_token(mini_batch_messages_list)
                ## get selected IDs of docs
                mini_batch_selector_inputs, mini_batch_selector_answers = self.get_inputs(mini_batch_token_with_mask, logits_processor)
                mini_batch_selector_answers_text = []
                for tem_i in range(len(mini_batch_selector_answers)):
                    temp_answer = self.tokenizer.decode(mini_batch_selector_answers[tem_i], skip_special_tokens=True)
                    mini_batch_selector_answers_text.append(temp_answer)
                mini_batch_selected_docs_ID = []
                for temp_i in range(len(mini_batch_selector_answers_text)):
                    answer = mini_batch_selector_answers_text[temp_i]
                    K_candidate_num = len(mini_batch_input_candidate_docs[temp_i])
                    number_answer = self.extract_digits(answer, K_candidate_num)

                    # 打开文件进行追加写入
                    selector_results_path = self.args.output_dir + '/context_selector.txt'
                    with open(selector_results_path, 'a') as file:
                        # 将列表转换为字符串并用空格连接，写入一行
                        # line = ' '.join(map(str, number_answer))
                        file.write(answer + '\n')

                    mini_batch_selected_docs_ID.append(number_answer)

                if step < CRITIC_STEP:
                    queries.extend(mini_batch_selector_inputs)
                    responses.extend(mini_batch_selector_answers)
                else:
                    if step % SELECTOR_STEP == 0:
                        queries.extend(mini_batch_selector_inputs)
                        responses.extend(mini_batch_selector_answers)

                # ****************************************** generator ******************************************
                ## construct input messages of generator
                mini_batch_messages_list = []
                for batch_i in range(len(mini_batch_input_questions)):
                    question = mini_batch_input_questions[batch_i]
                    top_docs = mini_batch_input_candidate_docs[batch_i]
                    selected_IDs = mini_batch_selected_docs_ID[batch_i]
                    generator_candidate_docs = []
                    for temp_id in selected_IDs:
                        generator_candidate_docs.append(top_docs[temp_id])
                    generator_messages = self.get_generator_messages(question, generator_candidate_docs)
                    mini_batch_messages_list.append(generator_messages)
                ## encode text to ids
                mini_batch_token_with_mask = self.trans_text_to_token(mini_batch_messages_list)
                ## get final answers
                mini_batch_generator_inputs, mini_batch_generator_answers = self.get_inputs(mini_batch_token_with_mask)

                if step < CRITIC_STEP:
                    queries.extend(mini_batch_generator_inputs)
                    responses.extend(mini_batch_generator_answers)
                else:
                    if step % GENERATOR_STEP == 0:
                        queries.extend(mini_batch_generator_inputs)
                        responses.extend(mini_batch_generator_answers)

                # ****************************************** rewards ******************************************
                ## get golden answers
                golden_answers = []
                for batch_i in range(len(mini_batch_input_questions)):
                    question = mini_batch_input_questions[batch_i]
                    try:
                        golden_answers.append(questions_golden_answers_dict[question.replace(" ", "")])
                    except KeyError:
                        golden_answers.append("")
                        print('KeyError: {}'.format(question))

                ## get predict answers
                predict_answers = []
                for response_ids in mini_batch_generator_answers:
                    predict_answer = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    predict_answer = predict_answer.replace("*", "")
                    predict_answers.append(predict_answer)
                
                # 保存预测答案和标准答案
                for temp_id in range(len(predict_answers)):
                    pred_ans = predict_answers[temp_id]
                    gold_ans = golden_answers[temp_id]
                    # 打开文件进行追加写入
                    generator_results_path = self.args.output_dir + '/context_generator.txt'
                    with open(generator_results_path, 'a') as file:
                        # 将列表转换为字符串并用空格连接，写入一行
                        file.write(pred_ans + '\t||\t' + gold_ans + '\n')

                mini_batch_rewards = self.get_rewards(predict_answers, golden_answers, reward_metric_name='f1')
                mini_batch_rewards_selctor_aindoc = self.get_selector_rewards(predict_answers, golden_answers, mini_batch_input_candidate_docs, mini_batch_input_questions, mini_batch_selector_answers_text)
                
                mini_batch_repeat_punish = self.get_selector_repeat_punish(mini_batch_selector_answers_text, mini_batch_input_candidate_docs)
                mini_batch_qr_punish = self.get_qr_punish(predict_answers, mini_batch_subquestions)
                mini_batch_generator_punish = self.get_generator_punish(predict_answers)  # 过长输出的punish

                # 不同模块的rewards
                if step < CRITIC_STEP:
                    rewards.extend(mini_batch_rewards+mini_batch_qr_punish)  # reward for qr
                    rewards.extend(mini_batch_rewards+mini_batch_repeat_punish)  # reward for selector
                    rewards.extend(mini_batch_rewards+mini_batch_generator_punish)  # reward for generator
                else:
                    if step % QR_STEP == 0 and step > 0:
                        rewards.extend(mini_batch_rewards+mini_batch_qr_punish)
                    if step % SELECTOR_STEP == 0:
                        rewards.extend(mini_batch_rewards+mini_batch_repeat_punish)
                    if step % GENERATOR_STEP == 0:
                        rewards.extend(mini_batch_rewards+mini_batch_generator_punish)

                reward_qr = (mini_batch_rewards+mini_batch_qr_punish).mean().item()
                reward_selctor_aindoc = mini_batch_rewards_selctor_aindoc.mean().item()
                reward_generator_final = mini_batch_rewards.mean().item()
                # 打开文件进行追加写入
                reward_qr_path = self.args.output_dir+'/reward_qr.txt'
                with open(reward_qr_path, 'a') as file:
                    # 将列表转换为字符串并用空格连接，写入一行
                    file.write(str(reward_qr) + '\n')
                # 打开文件进行追加写入
                reward_selctor_aindoc_path = self.args.output_dir+'/reward_selctor_aindoc.txt'
                with open(reward_selctor_aindoc_path, 'a') as file:
                    # 将列表转换为字符串并用空格连接，写入一行
                    file.write(str(reward_selctor_aindoc) + '\n')
                # 打开文件进行追加写入
                reward_generator_final_path = self.args.output_dir+'/reward_generator_final.txt'
                with open(reward_generator_final_path, 'a') as file:
                    # 将列表转换为字符串并用空格连接，写入一行
                    file.write(str(reward_generator_final) + '\n')

            rollout_end_time = time.time()
            if step < 3:
                print(step, 'rollout time consuming: {} seconds'.format(rollout_end_time - rollout_start_time))

            update_start_time = time.time()

            # Run PPO step
            self.model.train()

            # print('********begin step********')
            stats = self.step(queries, responses, rewards)
            # print('********end step********')

            update_end_time = time.time()
            if step < 3:
                print(step, 'update time consuming: {} seconds'.format(update_end_time - update_start_time))

            # 保证self.kl_ctl.value不小于某个值Min_beta
            Min_beta = 0.08
            if self.kl_ctl.value < Min_beta:
                self.kl_ctl.value = Min_beta

            self.tokenizer.padding_side = "left"  # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except Exception:
                    logger.warning("Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            self.callback_handler.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.args, self.state, self.control, logs)
                loss_meter.reset()
                reward_meter.reset()

            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(self.args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, self.state.global_step))
                )
                self.callback_handler.on_save(self.args, self.state, self.control)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)

    @override
    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    @override
    def create_scheduler(
        self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @torch.no_grad()
    def get_inputs(self, batch: Dict[str, "torch.Tensor"], logits_processor=[]) -> Tuple[List["torch.Tensor"], List["torch.Tensor"]]:
        r"""
        Generates model's responses given queries.
        """
        if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                batch[k] = v[:, start_index:]

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)
            
            if len(logits_processor) != 0:
                generate_output: "torch.Tensor" = unwrapped_model.generate(
                    generation_config=self.generation_config, logits_processor=logits_processor, max_new_tokens=29, **batch
                )
            elif len(logits_processor) == 0:
                generate_output: "torch.Tensor" = unwrapped_model.generate(
                    generation_config=self.generation_config, logits_processor=get_logits_processor(), max_new_tokens=100, **batch
                )

            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped_model, layernorm_params)

        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1) :].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_indexes) == 0:  # allow empty response
                response_length = 1
            elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:  # include eos token
                response_length = response_indexes[-1].item() + 2
            else:
                response_length = response_indexes[-1].item() + 1

            queries.append(query[i, query_start_index:])  # remove padding from left
            responses.append(response[i, :response_length])  # remove padding from right

        return queries, responses

    # @torch.no_grad()
    # def get_rewards(
    #     self,
    #     queries: List["torch.Tensor"],
    #     responses: List["torch.Tensor"],
    # ) -> List["torch.Tensor"]:
    #     r"""
    #     Computes scores using given reward model.

    #     Both inputs and outputs are put on CPU.
    #     """
    #     if self.finetuning_args.reward_model_type == "api":
    #         token_ids = [torch.cat((q, r), dim=-1).tolist() for q, r in zip(queries, responses)]
    #         messages = self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
    #         return get_rewards_from_server(self.reward_model, messages)

    #     batch: Dict[str, "torch.Tensor"] = self.prepare_model_inputs(queries, responses)
    #     unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)

    #     if self.finetuning_args.reward_model_type == "lora":
    #         replace_model(unwrapped_model, target="reward")
    #         reward_model = self.model
    #     else:
    #         reward_model = self.reward_model

    #     with unwrap_model_for_generation(reward_model, self.accelerator), self.amp_context:  # support bf16
    #         values: "torch.Tensor" = reward_model(**batch, return_dict=True, use_cache=False)[-1]

    #     if self.finetuning_args.reward_model_type == "lora":
    #         replace_model(unwrapped_model, target="default")

    #     rewards = values.gather(dim=-1, index=(batch["attention_mask"].sum(dim=-1, keepdim=True) - 1))
    #     return rewards.float().detach()  # use fp32 type

    @override
    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: "torch.Tensor",
        responses: "torch.Tensor",
        model_inputs: Dict[str, Any],
        return_logits: bool = False,
        response_masks: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        r"""
        Calculates model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with self.amp_context:  # support bf16
                logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @override
    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)  # must be called at all ranks
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)

        elif self.args.should_save:
            unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
            self._save(output_dir, state_dict=unwrapped_model.state_dict())
