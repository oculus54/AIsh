# AIsh

### AI-Powered Natural Language → Bash Command Generation
![QLoRA](https://img.shields.io/badge/QLoRA-4--Bit%20Fine--Tuning-8A2BE2?style=for-the-badge)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-FF6F00?style=for-the-badge)
![Transformers](https://img.shields.io/badge/Transformers-FFCC00?style=for-the-badge\&logo=huggingface\&logoColor=black)
![TRL](https://img.shields.io/badge/TRL-Fine--Tuning-FFCC00?style=for-the-badge\&logo=huggingface\&logoColor=black)
![llama.cpp](https://img.shields.io/badge/llama.cpp-Local%20Inference-000000?style=for-the-badge\&logo=github\&logoColor=white)

AIsh is a QLoRA fine-tuned language model based on **Qwen2.5-1.5B-Instruct**, trained to convert natural-language instructions into concise Bash commands.

The goal is simple: describe what you want to do in the terminal, and AIsh generates the corresponding Bash command without unnecessary explanation.

---

## Example


![Sample Output](examples\ssaish.png)

## Model

| Component              | Configuration                          |
| ---------------------- | -------------------------------------- |
| Base Model             | Qwen2.5-1.5B-Instruct                  |
| Fine-tuning            | QLoRA                                  |
| Quantization           | 4-bit NF4                              |
| Double Quantization    | Enabled                                |
| LoRA Rank              | 16                                     |
| LoRA Alpha             | 32                                     |
| LoRA Dropout           | 0.05                                   |
| Target Modules         | q, k, v, o, gate, up, down projections |
| Sequence Length        | 256 tokens                             |
| Training Epochs        | 5                                      |
| Learning Rate          | 2e-4                                   |
| Scheduler              | Cosine                                 |
| Optimizer              | Paged AdamW 8-bit                      |
| Effective Batch Size   | 16                                     |
| Gradient Checkpointing | Enabled                                |

The model was trained using QLoRA, keeping the base model quantized while training lightweight LoRA adapters.

---

## Dataset

AIsh was trained on the [`aelhalili/bash-commands-dataset`](https://huggingface.co/datasets/aelhalili/bash-commands-dataset).

The training pipeline removes duplicate `prompt → response` pairs before splitting the dataset into training and validation sets.

* Approximately **700 unique examples**
* **90% training / 10% validation**
* Split seed: `42`
* Short natural-language instructions paired with Bash commands

---

## Training

Training was performed in **Google Colab using an NVIDIA T4 GPU**.

The model uses:

* 4-bit NF4 quantization
* LoRA-based parameter-efficient fine-tuning
* 8-bit paged AdamW optimizer
* Gradient accumulation
* Gradient checkpointing
* Cosine learning-rate scheduling
* 3% warmup ratio

The training configuration was designed to fit a small instruction-following model and dataset within limited GPU memory.

---

## Results

### Training Metrics

| Epoch | Training Loss | Validation Loss | Mean Token Accuracy |
| ----: | ------------: | --------------: | ------------------: |
|     1 |        0.3034 |          0.2894 |              91.73% |
|     2 |        0.2489 |          0.2700 |              92.20% |
|     3 |        0.2065 |      **0.2670** |          **92.43%** |
|     4 |        0.1781 |          0.2738 |              92.44% |
|     5 |    **0.1680** |          0.2753 |          **92.44%** |

### Final Training Statistics

```text
Epochs:              5
Training steps:      200
Training loss:       0.2873
Final train loss:    0.1680
Final validation loss: 0.2753
Mean token accuracy: 92.44%
Training runtime:    ~23.8 minutes
GPU:                 NVIDIA T4
```

The validation loss reached its lowest recorded value around **epoch 3**, while token accuracy subsequently remained almost unchanged. This indicates that the model had largely converged on the small training dataset by that point.

> **Note:** Mean token accuracy measures token-level prediction accuracy. It should not be interpreted as exact Bash command accuracy or command execution success.

---

## Inference with Transformers

Load the trained LoRA adapter on top of the original Qwen model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct"
)

model = PeftModel.from_pretrained(
    base,
    "adapter/"
)

tokenizer = AutoTokenizer.from_pretrained("adapter/")
```

Use the same system instruction used during training:

```text
You are a helpful assistant that converts natural language instructions into a single correct Bash command. Reply with only the command, no explanation.
```

---

## llama.cpp

The LoRA adapter can also be converted to GGUF and used with `llama.cpp`.

```bash
python convert_lora_to_gguf.py \
    adapter/ \
    --outfile bash-commands-adapter.gguf \
    --outtype q8_0
```

Then pair the adapter with the original Qwen2.5-1.5B-Instruct GGUF model:

```bash
llama-cli \
    -m qwen2.5-1.5b-instruct-q8_0.gguf \
    --lora bash-commands-adapter.gguf \
    -cnv \
    -p "You are a helpful assistant that converts natural language instructions into a single correct Bash command. Reply with only the command, no explanation."
```

---

## Training Pipeline

```text
Natural Language Prompt
          │
          ▼
   Bash Commands Dataset
          │
          ▼
   Duplicate Removal
          │
          ▼
     90/10 Split
          │
          ▼
 Qwen2.5-1.5B-Instruct
          │
          ▼
      4-bit NF4
          │
          ▼
       QLoRA
          │
          ▼
   LoRA Adapter Training
          │
          ▼
    Fine-tuned Adapter
          │
          ▼
 Transformers / llama.cpp
          │
          ▼
      Bash Command
```

---

## Project Structure

```text
AIsh/
├── adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
│
├── notebook/
│   └── qlora_bash_commands_finetune.ipynb
│
├── examples/
│   └── sample_outputs.md
│
├── README.md
└── LICENSE
```

---

## Limitations

AIsh is trained on a relatively small, domain-specific dataset. Its output should therefore be treated as generated code rather than guaranteed-safe shell commands.

Potential failure cases include:

* Unsupported Bash operations
* Incorrect flags or arguments
* Environment-specific commands
* Destructive commands
* Commands requiring elevated privileges
* Complex multi-step shell workflows

**Do not blindly execute generated commands**, particularly commands involving `sudo`, `rm`, permissions, disks, networking, or system configuration.

---

## Future Improvements

Potential improvements include:

* Increasing dataset size and diversity
* Adding more complex Bash workflows
* Evaluating exact command-match accuracy
* Measuring command execution success
* Adding syntax validation
* Adding safety filtering for destructive commands
* Testing against unseen command patterns
* Comparing different QLoRA configurations
* Building an interactive terminal assistant around the model

---






