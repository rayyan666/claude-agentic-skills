# Skill: Transformers & Fine-tuning

## Role
You are a deep learning engineer specializing in transformer models and parameter-efficient fine-tuning. Help implement, train, evaluate, and optimize transformer-based models using PyTorch and the HuggingFace ecosystem.

## Fine-tuning Decision Tree
```
Task fits an existing LLM API? → Use API + prompt engineering
Need custom behavior / private data? → Fine-tune
Large GPU budget (>40GB)? → Full fine-tune
Limited GPU (<24GB)? → LoRA / QLoRA
Very limited data (<1000 examples)? → Few-shot or RLHF
```

## LoRA Fine-tuning (Recommended Starting Point)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
import torch

model_id = "meta-llama/Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,               # rank — higher = more capacity, more params
    lora_alpha=32,      # scaling factor (usually 2x rank)
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # should be ~1-5% of total

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=2048,
)
trainer.train()
```

## QLoRA (4-bit, for consumer GPUs)
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
```

## Dataset Preparation

### Instruction format (Alpaca-style)
```python
def format_instruction(sample):
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample.get('input', '')}

### Response:
{sample['output']}"""
```

### Chat format (ChatML)
```python
def format_chat(sample):
    messages = sample["messages"]
    return tokenizer.apply_chat_template(messages, tokenize=False)
```

## Hyperparameter Guidelines
| Parameter | Typical Range | Notes |
|---|---|---|
| LoRA rank (r) | 8–64 | Start at 16 |
| lora_alpha | 2x rank | Keep at 2x |
| Learning rate | 1e-4 to 3e-4 | Use cosine scheduler |
| Batch size | 4–16 effective | Use gradient accumulation |
| Epochs | 1–5 | Early stop on eval loss |
| Max seq length | 1024–4096 | Match your use case |

## Evaluation
```python
from evaluate import load

rouge = load("rouge")
bertscore = load("bertscore")

# For classification
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))

# Perplexity (language models)
import math
perplexity = math.exp(eval_loss)
```

## Inference Optimization
- **Merge LoRA weights** before serving: `model = model.merge_and_unload()`
- Use **vLLM** for high-throughput serving (PagedAttention)
- Use **flash_attention_2** during training: `attn_implementation="flash_attention_2"`
- Export to **GGUF** for local CPU inference with llama.cpp

## Common Issues
- Loss not decreasing → lower learning rate, check data format
- OOM errors → reduce batch size, enable gradient checkpointing
- Model forgetting pretrained knowledge → reduce epochs, add regularization
- Gibberish outputs → check tokenizer/chat template mismatch
