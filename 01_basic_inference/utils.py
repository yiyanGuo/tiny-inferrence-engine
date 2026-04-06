import torch
import time

def run_benchmark(func, model, tokenizer, prompts, device, max_new_tokens):
    if device == "cuda":
        torch.cuda.synchronize()  # 确保所有 CUDA 操作完成
        torch.cuda.reset_peak_memory_stats()  # 重置 CUDA 内存统计
        torch.cuda.empty_cache()  # 清空 CUDA 缓存
        torch.cuda.synchronize()  # 再次确保所有 CUDA 操作完成
    
    start_time = time.time()

    func(model, tokenizer, prompts, device, max_new_tokens)

    if device == "cuda":
        torch.cuda.synchronize()  # 确保所有 CUDA 操作完成
    end_time = time.time()

    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # 转换为 MB
        print(f"[{func.__name__}] Peak GPU Memory Usage: {peak_memory:.2f} MB")
    print(f"[{func.__name__}] Total Inference Time: {end_time - start_time:.2f} seconds")

