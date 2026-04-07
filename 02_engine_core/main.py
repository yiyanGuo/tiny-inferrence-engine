from client import Client

def main():
    model_name = "/root/autodl-tmp/Qwen3-8b"  # 替换为实际模型路径
    client = Client(model_name)

    prompts = ["Who are you?", "What is your name?"]
    client.submit_request(prompts)

    client.generate()

if __name__ == '__main__':
    main()