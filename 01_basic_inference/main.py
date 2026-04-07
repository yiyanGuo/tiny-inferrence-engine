from engine import Engine
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt


def main():
    model_name = "/root/autodl-tmp/Qwen3-8b"  # 替换为实际模型路径
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = ["Who are you?", "What is your name?"]

    # 初始化推理引擎
    engine = Engine(model, tokenizer, device)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    max_new_tokens = [4096]
    basic_inference_results = []
    kv_cache_inference_results = []

    for max_tokens in max_new_tokens:
        # print(f"Running basic inference benchmark with max_new_tokens={max_tokens}...")
        # basic_res = engine.naive_generate(inputs, max_new_tokens=max_tokens, profile=True)
        print(f"Running KV cache inference benchmark with max_new_tokens={max_tokens}...")
        kv_cache_res = engine.my_generate(inputs, max_new_tokens=max_tokens, profile=True)

    engine.logger.print()
    # # 可视化结果
    # plt.figure(figsize=(12, 6))
    # plt.plot(max_new_tokens, [res[1]["inference_time"] for res in basic_inference_results], label="Basic Inference", marker='o')
    # plt.plot(max_new_tokens, [res[1]["inference_time"] for res in kv_cache_inference_results], label="KV Cache Inference", marker='o')
    # plt.xlabel("Max New Tokens")
    # plt.yscale("log")
    # plt.ylabel("Inference Time (s)")
    # plt.title("Inference Time vs Max New Tokens")
    # plt.legend()
    # plt.savefig("inference_time_comparison.png")


    
if __name__ == '__main__':
    main()