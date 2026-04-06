from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def basic_forward(model, inputs, kv_cache):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    dtype = model.dtype # 获取模型权重精度

    # 1. 准备 Position IDs
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

    # 2. 构造 4D 掩码 (与模型精度对齐)
    causal_mask = torch.full((seq_len, seq_len), fill_value=torch.finfo(dtype).min, device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)

    padding_mask = (1.0 - attention_mask.to(dtype)) * torch.finfo(dtype).min
    padding_mask = padding_mask.view(batch_size, 1, 1, seq_len)
    combined_mask = causal_mask + padding_mask

    # 3. 获取 RoPE Embeddings
    # 这里的关键是：Qwen2 的 rotary_emb 期望输入一个张量来推导数据类型
    # 我们直接取第 0 个 batch 的输出，确保它是 (seq_len, head_dim) 格式
    cos, sin = model.model.rotary_emb(model.model.embed_tokens(input_ids), position_ids)
    # position_embeddings 必须是元组且符合 (seq, dim) 或 (batch, 1, seq, dim)
    position_embeddings = (cos, sin)

    # 4. Embedding
    hidden_states = model.model.embed_tokens(input_ids)

    # 5. Layers
    for i, layer in enumerate(model.model.layers):
        outputs = layer(
            hidden_states,
            attention_mask=combined_mask,
            position_ids=position_ids,
            past_key_value=kv_cache[i],
            position_embeddings=position_embeddings, # 传入预计算的旋转编码
            use_cache=True
        )
        hidden_states = outputs[0]
        if kv_cache is not None:
            kv_cache[i] = outputs[1]

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits

def test():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # print(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = ["Who are you?", "What is your name?"]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    kv_cache = [None] * len(model.model.layers)  # 初始化 KV cache

    logits = basic_forward(model, inputs, kv_cache)
    print(logits.shape)  # 应该是 (batch_size, seq_len, vocab_size)

if __name__ == "__main__":
    test()