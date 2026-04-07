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

        self.kv_cache = {}  # {request_id: kv_cache}
        self.steps = 0
        self.max_new_tokens = 200
        # # TODO
        self.scheduler = Scheduler()
        # self.kv_cache_manager = None
    
    def submit_request(self, prompt, request_id):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        request = Request(
            request_id=request_id,
            request_status=RequestStatus.PREFILLING,
            inputs=inputs
        )
        self.scheduler.submit_request(request)
    
    @torch.inference_mode()
    def _prefill(self, requests: list[Request]):
        # batch prefill
        finished_request_ids = []
        for req in requests:
            outputs = self.model(input_ids=req.inputs["input_ids"], attention_mask=req.inputs["attention_mask"], use_cache=True)
            self.kv_cache[req.request_id] = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            req.inputs["input_ids"] = torch.cat([req.inputs["input_ids"], next_token], dim=-1)
            req.inputs["attention_mask"] = torch.cat([req.inputs["attention_mask"]  , torch.ones_like(next_token)], dim=-1)
            req.request_status = RequestStatus.DECODING
        
        return finished_request_ids
    
    @torch.inference_mode()
    def _decode(self, requests: list[Request]):
        finished_request_ids = []
        for req in requests:
            kv_cache = self.kv_cache[req.request_id]
            current_input_ids = req.inputs["input_ids"][:, -1].unsqueeze(-1)
            attention_mask = req.inputs["attention_mask"]
            outputs = self.model(input_ids=current_input_ids, attention_mask=attention_mask, past_key_values=kv_cache, use_cache=True)
            self.kv_cache[req.request_id] = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            if next_token.item() == self.tokenizer.eos_token_id or req.inputs["input_ids"].shape[-1] >= self.max_new_tokens:
                req.request_status = RequestStatus.FINISHED
                finished_request_ids.append(req.request_id)
                continue
            req.inputs["input_ids"] = torch.cat([req.inputs["input_ids"], next_token], dim=-1)
            req.inputs["attention_mask"] = torch.cat([req.inputs["attention_mask"]  , torch.ones_like(next_token)], dim=-1)
        
        return finished_request_ids
    @torch.inference_mode()
    def step(self):
        prefilling_reqs, decoding_reqs = self.scheduler.schedule()
        finished_request_ids = []
        # prefill
        if prefilling_reqs:
            finished_request_ids.extend(self._prefill(prefilling_reqs))
        # decoding
        if decoding_reqs:
            finished_request_ids.extend(self._decode(decoding_reqs))
        # update scheduler
        self.scheduler.update_after_step(finished_request_ids)
    
    def run(self):
        print("Starting engine loop...")

        while True:
            if not self.scheduler.has_active_requests():
                break

            self.step()
            self.steps += 1

            if self.steps % 10 == 0:
                print(f"Step {self.steps}")

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