import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from log import Logger
from scheduler import Scheduler
from request import Request, RequestStatus

class Engine:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.logger = Logger()

        self.queue = []
        self.kv_cache = {}  # {request_id: kv_cache}
        self.steps = 0
        self.max_new_tokens = 200
        # # TODO
        # self.scheduler = Scheduler()
        # self.kv_cache_manager = None
    
    def submit_request(self, prompts, request_id):
        for i, prompt in enumerate(prompts):
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            request = Request(
                request_id=f"{request_id}_{i}", 
                prompt=prompt, 
                request_status=RequestStatus.PREFILLING, 
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            
            # TODO： 改用scheduler管理请求
            self.queue.append(request)
    
    def _get_active_requests(self):
        return [req for req in self.queue if req.request_status != RequestStatus.FINISHED and req.request_status != RequestStatus.ABORTED]

    def _get_prefilling_requests(self):
        return [req for req in self.queue if req.request_status == RequestStatus.PREFILLING]

    def _get_decoding_requests(self):
        return [req for req in self.queue if req.request_status == RequestStatus.DECODING]
    
    @torch.inference_mode()
    def step(self):
        # prefill
        prefilling_reqs = self._get_prefilling_requests()
        for req in prefilling_reqs:
            outputs = self.model(input_ids=req.input_ids, attention_mask=req.attention_mask, use_cache=True)
            self.kv_cache[req.request_id] = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            req.input_ids = torch.cat([req.input_ids, next_token], dim=-1)
            req.attention_mask = torch.cat([req.attention_mask, torch.ones_like(next_token)], dim=-1)
            req.request_status = RequestStatus.DECODING
        
        # decode
        decoding_reqs = self._get_decoding_requests()
        for req in decoding_reqs:
            kv_cache = self.kv_cache[req.request_id]
            current_input_ids = req.input_ids[:, -1].unsqueeze(-1)
            attention_mask = req.attention_mask
            outputs = self.model(input_ids=current_input_ids, attention_mask=attention_mask, past_key_values=kv_cache, use_cache=True)
            self.kv_cache[req.request_id] = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            if next_token.item() == self.tokenizer.eos_token_id or req.input_ids.shape[-1] >= self.max_new_tokens:
                req.request_status = RequestStatus.FINISHED
                continue
            req.input_ids = torch.cat([req.input_ids, next_token], dim=-1)
            req.attention_mask = torch.cat([req.attention_mask, torch.ones_like(next_token)], dim=-1)
    
    def run(self):
        print("Starting engine loop...")

        while True:
            active_reqs = self._get_active_requests()
            if not active_reqs:
                break

            self.step()
            self.steps += 1

            if self.steps % 10 == 0:
                print(f"Step {self.steps}, {len(active_reqs)} active requests")

        print(f"All requests finished. After {self.steps} steps.")





        


    
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