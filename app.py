import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

#BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct.gguf" add your gguf file here  
ADAPTER_PATH = "./adapter"

print("Loading AIsh...")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Load QLoRA adapter
model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

model.eval()

print("AIsh loaded.\n")

SYSTEM_PROMPT = """You are AIsh, an AI assistant that converts natural language
instructions into a single Bash command.

Rules:
- Return only the Bash command.
- Do not explain the command.
- Do not use Markdown code blocks.
- Generate a safe and appropriate Bash command.
"""


def generate_command(user_input):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Remove input tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return response


def main():
    print("AIsh: Natural Language → Bash")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You > ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting AIsh.")
            break

        if not user_input:
            continue

        try:
            command = generate_command(user_input)

            print(f"\nBash > {command}\n")

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()