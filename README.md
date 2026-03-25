# Claude Skills for Agentic AI & Deep Learning Engineers

> A curated set of Claude skill files for AI/ML engineers building agentic systems and deep learning pipelines. Upload these to your Claude environment to supercharge your workflow.

---

##  What's Included

| Skill File | Purpose |
|---|---|
| `prompt-engineering.md` | Master system prompts, few-shot, CoT, XML structuring |
| `agentic-loops.md` | Build ReAct agents, tool use, LangChain/LangGraph patterns |
| `rag-vector-stores.md` | RAG pipelines, embeddings, FAISS/Pinecone/Weaviate |
| `transformers-finetuning.md` | HuggingFace, LoRA/QLoRA, PEFT, PyTorch training loops |
| `mlops-observability.md` | LangSmith, W&B, CI/CD, evaluation, deployment |
| `guardrails-alignment.md` | Output validation, hallucination detection, safety patterns |

---

##  How to Use

### Option 1: Upload to Claude.ai Projects
1. Go to [claude.ai](https://claude.ai) → **Projects**
2. Create a new project (e.g., `Agentic AI Dev`)
3. Click **Add content** → upload any `.md` skill files
4. Claude will use them as persistent context in all project conversations

### Option 2: Paste as System Prompt
Copy the content of any skill file and paste it into:
- Claude API `system` parameter
- Your LangChain `SystemMessage`
- Any LLM orchestration framework's system prompt field

### Option 3: Use All Skills Together
For a comprehensive setup, combine all files into one system prompt:
```python
import os

skills_dir = "./skills"
combined = ""
for fname in os.listdir(skills_dir):
    if fname.endswith(".md"):
        with open(os.path.join(skills_dir, fname)) as f:
            combined += f.read() + "\n\n---\n\n"

# Use combined as your system prompt
```

---

##  Skill Breakdown

### 1. Prompt Engineering & Context Design
Covers structured prompting techniques optimized for Claude and other LLMs — including XML tagging, role prompting, chain-of-thought scaffolding, and few-shot example design.

### 2. Agentic Loops & Tool Use
Patterns for building reliable autonomous agents: ReAct loops, plan-and-execute, multi-agent coordination, and tool/function calling best practices.

### 3. RAG & Vector Stores
End-to-end retrieval-augmented generation: chunking strategies, embedding models, vector DB setup, hybrid search, and reranking.

### 4. Transformers & Fine-tuning
PyTorch-first deep learning patterns: HuggingFace Trainer, LoRA/QLoRA with PEFT, dataset preparation, evaluation metrics, and inference optimization.

### 5. MLOps & Agent Observability
Production ML practices: experiment tracking with W&B, agent tracing with LangSmith, model versioning, CI/CD pipelines, and A/B evaluation frameworks.

### 6. Guardrails & Alignment
Safety-first engineering: output validators, hallucination detection heuristics, Constitutional AI principles, red-teaming checklists, and responsible deployment patterns.

---

##  Requirements

These skills are plain Markdown — no dependencies required. They work with:
- Claude (claude.ai, API)
- GPT-4 / GPT-4o
- Any LLM that accepts a system prompt

---

##  Contributing

PRs welcome! If you've built a skill that improved your agentic AI workflow, open a pull request with:
- The `.md` skill file in `/skills/`
- A one-paragraph description added to this README

---

##  License

MIT — free to use, modify, and distribute.

---

*Built for AI/ML engineers working on the frontier of agentic systems and deep learning.*
