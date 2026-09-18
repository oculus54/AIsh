sh-commands-qlora

A QLoRA fine-tune of [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) that
converts natural-language instructions into bash commands.

## Dataset
[`aelhalili/bash-commands-dataset`](https://huggingface.co/datasets/aelhalili/bash-commands-dataset) (MIT) — ~700 unique prompt→command pairs after deduplication.

## Training
- Base model: Qwen2.5-1.5B-Instruct, loaded in 4-bit (QLoRA)
- LoRA: rank 16, alpha 32, dropout 0.05, targeting q/k/v/o/gate/up/down projections
- 5 epochs, batch size 4 (grad accum 4), lr 2e-4, cosine schedule
- Trained in Colab on a T4 — notebook in `notebook/`

Only the LoRA adapter (`adapter/`) is included here — a few MB, not the full model.
It's meant to be loaded on top of the original base model.

## Usage

### With transformers + peft
\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "adapter/")
tokenizer = AutoTokenizer.from_pretrained("adapter/")
\`\`\`

### With llama.cpp
Convert the adapter to a LoRA GGUF and pair it with the base model's GGUF at inference time:
\`\`\`bash
python convert_lora_to_gguf.py adapter/ --outfile bash-commands-adapter.gguf --outtype q8_0
llama-cli -m qwen2.5-1.5b-instruct-q8_0.gguf --lora bash-commands-adapter.gguf -cnv \
  -p "You are a helpful assistant that converts natural language instructions into a single correct Bash command. Reply with only the command, no explanation."
\`\`\`

## Examples
See [`examples/sample_outputs.md`](examples/sample_outputs.md).
