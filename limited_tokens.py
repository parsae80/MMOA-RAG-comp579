import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessorList, LogitsProcessor


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


def get_allowed_tokens(model, tokenizer, allowed_tokens=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):

    # 定义允许的token
    allowed_token_ids = tokenizer.convert_tokens_to_ids(allowed_tokens)
    eos_token_id = tokenizer.eos_token_id
    allowed_token_ids.append(eos_token_id)

    # 创建LogitsProcessor
    logits_processor = LogitsProcessorList([
        AllowedTokensLogitsProcessor(allowed_token_ids)
    ])

    return logits_processor


# # 加载模型和分词器
# generator_model_path = '/root/paddlejob/workspace/env_run/rag/models/LLM-Research/Meta-Llama-3-8B-Instruct'
# model = AutoModelForCausalLM.from_pretrained(
#     generator_model_path,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(generator_model_path)

# # 生成文本
# input_text = "5 - 1 = ? give me the answer number directly."
# input_ids = tokenizer.encode(input_text, return_tensors='pt')
# outputs = model.generate(
#         input_ids=input_ids,
#         max_new_tokens=50,
#         do_sample=False,
#         pad_token_id=tokenizer.eos_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#         logits_processor=get_allowed_tokens(model, tokenizer),
#         # eos_token_id=eos_token_id
# )

# # 解码并打印生成的文本
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(generated_text)
# print(len(generated_text.split('.')[-1]))