from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import time


def use_pipeline():
    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device=0 if torch.cuda.is_available() else -1)

    prompt = "hello, I am"
    output = pipe(prompt, max_new_tokens=20, do_sample=False)

    print(output[0]['generated_text'])

# 直接使用模型进行推理, 不使用 pipeline, 以便更清晰地展示推理过程
def basic_inference(model, inputs, max_new_tokens: int = 20):

    input_ids = inputs["input_ids"] # 形状为 (batch_size, seq_len), 其中 seq_len 是 prompts 中最长的文本的 token 数量
    attention_mask = inputs["attention_mask"] # 形状为 (batch_size, seq_len), 用于指示模型哪些位置是有效的输入, 哪些位置是 padding
    start_time = time.time()
    for _ in range(max_new_tokens): # 生成 max_new_tokens 个 token，为了简洁没有设置eof token, 实际使用中可以根据需要设置
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # 提取最后一个位置的 logits
        next_token_logits = outputs.logits[:, -1, :]
        # 选择概率最高的 token, 返回概率最大的索引
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        # 将生成的 token 添加到 input_ids 中
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        # 同时更新 attention_mask, 因为我们添加了一个新的 token
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
    end_time = time.time()

    generated_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    return {"generated_texts": generated_texts, "inference_time": end_time - start_time}

def kv_cache_inference(model, inputs, max_new_tokens: int = 20):

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    kv_cache = None
    generated_ids = input_ids

    start_time = time.time()
    # prefill
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    
    kv_cache = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

    generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)

    # decoding loop
    for _ in range(max_new_tokens - 1):
        current_input_ids = next_tokens

        attention_mask = torch.cat([attention_mask, torch.ones_like(current_input_ids)], dim=-1)

        with torch.no_grad():
            outputs = model(input_ids=current_input_ids, attention_mask=attention_mask, past_key_values=kv_cache, use_cache=True)
        
        kv_cache = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)

    
    end_time = time.time()

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return {"generated_texts": generated_texts, "inference_time": end_time - start_time}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = ["Who are you?", "What is your name?"]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    kv_cache_inference(model, inputs, max_new_tokens=20)