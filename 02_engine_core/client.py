from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from engine import Engine

class Client:
    def __init__(self, model_name:str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        self.engine = Engine(self.model, self.tokenizer, device)

        self.user_request_id_counter = 0


    def submit_request(self, prompts):
        user_request_id = f"userequest_{self.user_request_id_counter}"
        self.engine.submit_request(prompts, user_request_id)
        self.user_request_id_counter += 1


    def generate(self):
        self.engine.run()

