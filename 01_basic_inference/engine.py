import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from log import Logger

class Engine:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.logger = Logger()
    
    @torch.inference_mode()
    def generate(self, prompts, version:str, max_new_tokens=100, profile=False, ):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        if version == "naive":
            return self.naive_generate(inputs, max_new_tokens, profile=profile)
        elif version == "my":
            return self.my_generate(inputs, max_new_tokens, profile=profile)

    @torch.inference_mode()
    def naive_generate(self, inputs, max_new_tokens=100, use_cache=False, do_sample=False, profile=False):
        if profile:
            log = self.logger.profile("naive_generate")
            log.start()
        outtputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=use_cache, do_sample=do_sample)
        if profile:
            log.end()
        return outtputs
    
    
    @torch.inference_mode()
    def my_generate(self, inputs, max_new_tokens=100, profile=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        kv_cache = None
        generated_ids = input_ids

        # prefill
        if profile:
            log_prefill = self.logger.profile("my_generate_prefill")
            log_prefill.start()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

        kv_cache = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        if profile:
            log_prefill.end()
        # decoding loop
        if profile:
            log_decoding = self.logger.profile("my_generate_decoding")
            log_decoding.start()
        for _ in range(max_new_tokens - 1):
            current_input_ids = next_token

            attention_mask = torch.cat([attention_mask, torch.ones_like(current_input_ids)], dim=-1)

            outputs = self.model(input_ids=current_input_ids, attention_mask=attention_mask, past_key_values=kv_cache, use_cache=True)

            kv_cache = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        if profile:
            log_decoding.end()
        return generated_ids