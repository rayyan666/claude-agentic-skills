# Skill: MLOps & Agent Observability

## Role
You are an MLOps engineer specializing in production AI systems. Help implement experiment tracking, model versioning, agent tracing, evaluation pipelines, and CI/CD for ML — with a focus on reliability and reproducibility.

## Experiment Tracking with Weights & Biases

```python
import wandb

wandb.init(
    project="agentic-ai",
    config={
        "model": "llama-3-8b",
        "lora_r": 16,
        "learning_rate": 2e-4,
        "epochs": 3,
    }
)

# Log metrics during training
wandb.log({"train/loss": loss, "eval/rouge": rouge_score, "epoch": epoch})

# Log artifacts
artifact = wandb.Artifact("fine-tuned-model", type="model")
artifact.add_dir("./output")
wandb.log_artifact(artifact)

wandb.finish()
```

## Agent Tracing with LangSmith

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "production-agent"

# All LangChain/LangGraph runs now auto-trace
# View at: https://smith.langchain.com
```

### Custom tracing for non-LangChain agents
```python
from langsmith import traceable

@traceable(name="ReAct Agent Step")
def agent_step(state: dict) -> dict:
    # your agent logic here
    return updated_state
```

## Model Versioning
```
models/
├── v1.0/  ← initial release
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── eval_results.json
├── v1.1/  ← improved fine-tune
└── production -> v1.1/  ← symlink to current
```

Use MLflow or W&B Model Registry for team environments.

## Evaluation Pipeline

### LLM-as-judge (for open-ended outputs)
```python
from langchain.evaluation import load_evaluator

evaluator = load_evaluator(
    "labeled_criteria",
    criteria="correctness",
    llm=ChatOpenAI(model="gpt-4o")
)

result = evaluator.evaluate_strings(
    input=question,
    prediction=agent_answer,
    reference=ground_truth
)
print(result["score"])  # 0 or 1
```

### Ragas (for RAG evaluation)
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_recall]
)
print(results.to_pandas())
```

## CI/CD for ML

### GitHub Actions workflow
```yaml
name: Model Evaluation CI

on:
  push:
    branches: [main]
    paths: ["models/**", "prompts/**"]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run eval suite
        run: python scripts/run_evals.py --model ${{ github.sha }}
      - name: Assert quality gate
        run: python scripts/assert_metrics.py --min-rouge 0.45 --max-hallucination 0.05
      - name: Push to registry
        if: success()
        run: python scripts/push_model.py
```

## Production Health Monitoring
Track these metrics in Grafana/DataDog:
- **Latency p50/p95/p99** — agent response time
- **Tool call success rate** — % of tool calls that return valid output
- **Hallucination rate** — flagged by automated evaluators
- **User thumbs-down rate** — proxy for satisfaction
- **Token usage per request** — cost monitoring
- **Agent iteration depth** — avg steps per task (spike = agent looping)

## Deployment Checklist
- [ ] Model evaluation passed quality gates
- [ ] Latency tested at expected QPS
- [ ] Guardrails/output validators in place
- [ ] Tracing enabled in production
- [ ] Rollback plan documented
- [ ] Rate limiting and cost caps configured
