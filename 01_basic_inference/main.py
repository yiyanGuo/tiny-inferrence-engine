from inference import use_pipeline, basic_inference, kv_cache_inference
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt


def run_benchmark(func, model, inputs, max_new_tokens, device="cpu"):
    if device == "cuda":
        torch.cuda.synchronize()  # 确保所有 CUDA 操作完成
        torch.cuda.reset_peak_memory_stats()  # 重置 CUDA 内存统计
        torch.cuda.empty_cache()  # 清空 CUDA 缓存
        torch.cuda.synchronize()  # 再次确保所有 CUDA 操作完成
    
    start_time = time.time()

    func(model, inputs, max_new_tokens)

    if device == "cuda":
        torch.cuda.synchronize()  # 确保所有 CUDA 操作完成
    end_time = time.time()

    profile_res = {"inference_time": end_time - start_time}
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # 转换为 MB
        profile_res["peak_memory"] = peak_memory
    
    return profile_res

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = ["Who are you?", "What is your name?"]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    max_new_tokens = range(100, 500, 50)
    basic_inference_results = []
    kv_cache_inference_results = []

    for max_tokens in max_new_tokens:
        print(f"Running basic inference benchmark with max_new_tokens={max_tokens}...")
        basic_res = run_benchmark(basic_inference, model, inputs, max_tokens, device)
        basic_inference_results.append((max_tokens, basic_res))

        print(f"Running KV cache inference benchmark with max_new_tokens={max_tokens}...")
        kv_cache_res = run_benchmark(kv_cache_inference, model, inputs, max_tokens, device)
        kv_cache_inference_results.append((max_tokens, kv_cache_res))

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.plot(max_new_tokens, [res[1]["inference_time"] for res in basic_inference_results], label="Basic Inference", marker='o')
    plt.plot(max_new_tokens, [res[1]["inference_time"] for res in kv_cache_inference_results], label="KV Cache Inference", marker='o')
    plt.xlabel("Max New Tokens")
    plt.yscale("log")
    plt.ylabel("Inference Time (s)")
    plt.title("Inference Time vs Max New Tokens")
    plt.legend()
    plt.savefig("inference_time_comparison.png")


    
if __name__ == '__main__':
    main()