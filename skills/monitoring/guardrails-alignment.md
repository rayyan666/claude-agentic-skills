# Skill: Guardrails & Alignment

## Role
You are an AI safety and alignment engineer. Help design robust guardrails, output validators, hallucination detection systems, and responsible deployment patterns for production LLM and agentic AI systems.

## Defense-in-Depth Model

```
User Input → [Input Guard] → Agent/LLM → [Output Guard] → [Fact Check] → Response
                ↓ blocked                     ↓ blocked          ↓ flagged
             Safe refusal               Safe fallback        Human review
```

## Input Guardrails

### Prompt injection detection
```python
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "forget your system prompt",
    "you are now",
    "act as if you have no restrictions",
    "DAN mode",
    "jailbreak",
]

def detect_injection(user_input: str) -> bool:
    lower = user_input.lower()
    return any(p in lower for p in INJECTION_PATTERNS)
```

### Topic/scope filtering
```python
from anthropic import Anthropic

client = Anthropic()

def classify_intent(user_input: str, allowed_topics: list[str]) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        system=f"Classify the user's request. Allowed topics: {allowed_topics}. Return JSON: {{\"allowed\": true/false, \"topic\": \"...\"}}",
        messages=[{"role": "user", "content": user_input}]
    )
    return json.loads(response.content[0].text)
```

## Output Guardrails

### Structured output validation (Pydantic)
```python
from pydantic import BaseModel, validator
from typing import Optional

class AgentOutput(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    requires_human_review: bool = False

    @validator("confidence")
    def confidence_range(cls, v):
        assert 0.0 <= v <= 1.0, "Confidence must be between 0 and 1"
        return v

    @validator("answer")
    def no_pii(cls, v):
        import re
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", v):  # SSN pattern
            raise ValueError("Output contains potential PII")
        return v
```

### Hallucination detection
```python
def check_faithfulness(answer: str, source_docs: list[str], threshold=0.7) -> dict:
    """Check if answer is grounded in source documents using NLI."""
    from transformers import pipeline
    nli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    context = " ".join(source_docs[:3])
    result = nli(answer, candidate_labels=["supported", "not supported"], hypothesis_template=f"Based on: {context[:500]}, this statement is {{}}")
    
    score = result["scores"][result["labels"].index("supported")]
    return {"faithful": score >= threshold, "score": score}
```

## Constitutional AI Principles (for Claude-based agents)
When prompting Claude agents, embed these constraints:
```
Core principles for this agent:
1. Be honest — never fabricate facts, citations, or data
2. Be helpful — prioritize the user's actual goal, not just the literal request
3. Flag uncertainty — say "I'm not sure" rather than guessing
4. Preserve human oversight — escalate ambiguous or high-stakes decisions
5. Minimize harm — refuse requests that could cause direct harm, even if technically possible
```

## Red-Teaming Checklist
Before deploying an agent, test these adversarial scenarios:
- [ ] Prompt injection via user messages
- [ ] Prompt injection via tool outputs (indirect injection)
- [ ] Attempts to exfiltrate system prompt
- [ ] Requests to take irreversible actions (delete, send, publish)
- [ ] Inputs designed to trigger hallucination (fake citations, invented facts)
- [ ] Jailbreak attempts (roleplay, hypothetical framing)
- [ ] Data poisoning via RAG (adversarial documents in the vector store)
- [ ] Long context confusion (important instructions buried)

## Responsible Deployment Patterns

### Human-in-the-loop for high-stakes actions
```python
HIGH_RISK_ACTIONS = ["send_email", "delete_record", "execute_payment", "publish_content"]

def execute_tool(tool_name: str, args: dict) -> dict:
    if tool_name in HIGH_RISK_ACTIONS:
        approval = request_human_approval(tool_name, args)
        if not approval:
            return {"status": "cancelled", "reason": "User did not approve"}
    return tools[tool_name](**args)
```

### Audit logging
```python
import json, datetime

def log_agent_action(session_id, action_type, input_data, output_data, user_id):
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "session_id": session_id,
        "user_id": user_id,
        "action_type": action_type,
        "input_hash": hash(str(input_data)),  # don't log PII directly
        "output_summary": str(output_data)[:200],
    }
    # Write to append-only audit log
    with open("audit.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

## Safety Metrics to Track
- **Refusal rate** — % of inputs correctly blocked (too high = over-restrictive)
- **False positive rate** — safe inputs incorrectly blocked
- **Hallucination rate** — outputs not grounded in source material
- **PII leak rate** — sensitive data appearing in outputs
- **Escalation rate** — requests routed to human review
