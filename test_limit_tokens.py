import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessorList, LogitsProcessor

class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)
    
    def __call__(self, input_ids, scores):
        # Set the logits of unwanted tokens to a very low value
        mask = torch.full(scores.shape, float('-inf'))
        mask[:, list(self.allowed_token_ids)] = 0
        scores = scores + mask
        return scores

# 加载模型和分词器
generator_model_path = '/root/paddlejob/workspace/env_run/rag/models/LLM-Research/Meta-Llama-3-8B-Instruct'
print('loading generator model')
model = AutoModelForCausalLM.from_pretrained(
    generator_model_path,
    torch_dtype="auto",
    device_map="auto"
)
# model.eval()
# model = model.to(torch.float16)
tokenizer = AutoTokenizer.from_pretrained(generator_model_path)

# 定义允许的token
allowed_tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# allowed_tokens = ['Document 0', 'Document 1', 'Document 2', 'Document 3', 'Document 4', 'Document 5', 'Document 6', 'Document 7', 'Document 8', 'Document 9']
allowed_token_ids = tokenizer.convert_tokens_to_ids(allowed_tokens)
# eos_token_id = tokenizer.eos_token_id
# allowed_token_ids.append(eos_token_id)
print('allowed_token_ids: ', allowed_token_ids)

# 创建LogitsProcessor
logits_processor = LogitsProcessorList([
    AllowedTokensLogitsProcessor(allowed_token_ids)
])

# 生成文本
input_text = "Describe the location of Beijing."
input_ids = tokenizer.encode(input_text, return_tensors='pt')
outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=logits_processor,
        # eos_token_id=eos_token_id
)

# 解码并打印生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
print(len(generated_text.split('?')[-1]))