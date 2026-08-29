# Lab 27 — Human-in-the-Loop (HITL) Agent System

A LangGraph workflow that assesses customer churn risk, proposes a retention
action, and routes that action either to automatic execution or to a human
reviewer — depending on hard policy rules first and confidence second.

The point of the architecture: **a high confidence score is not authorisation.**
Policy decides what needs a human; confidence only decides among the things
policy already allows the agent to do alone.

## Install and run

```bash
pip install -r requirements.txt
```

```bash
streamlit run src/app.py      # the approval console
python -m src.graph           # smoke test: show where each customer routes
```

Tested on Python 3.11.9, langgraph 1.2.10, streamlit 1.62.0, pydantic 2.13.4.

## Files

| File | Contents |
|---|---|
| [src/models.py](src/models.py) | `AuditEntry` (Pydantic) and the append-only audit log I/O |
| [src/graph.py](src/graph.py) | `GraphState`, the three nodes, `route_action`, graph compilation |
| [src/app.py](src/app.py) | Streamlit console: action card, Approve / Reject / Edit, resume |
| [audit_log.json](audit_log.json) | The audit trail (created on first run) |
| [REFLECTION.md](REFLECTION.md) | Answers to the three reflection questions |

## Flow

```
customer_id
     |
     v
evaluate_customer          -> proposed_action, confidence_score, reasoning
     |
     v
route_action               -> hard rule checked BEFORE confidence
     |
     +--------------------------------+
     | low risk & conf >= 0.85        | hard rule OR conf < 0.85
     v                                v
execute_low_risk_action     [ INTERRUPT BEFORE ]
     |                      execute_high_risk_action
     |                                |
     |                      approve / edit -> execute
     |                      reject         -> abort
     |                                |
     +----------------+---------------+
                      v
              append to audit_log.json -> END
```

## Routing rules

Resolved in `classify_route()` in [src/graph.py](src/graph.py), in this order:

1. **Policy override** — `proposed_action in HIGH_RISK_ACTIONS` → human review.
   Evaluated before the confidence score is even read, so no score can buy a
   bypass. `increase_credit_limit` at 0.99 still stops for a human.
2. **Auto-execute** — low-risk action and `confidence >= 0.85` → run it.
3. **Escalate** — anything else → human review.

`classify_route()` returns both the next node and a `route_reason`, so the
reviewer sees *why* an item reached them (`policy_override` vs `low_confidence`).
`route_action()` and `evaluate_customer()` both call it, so the routing decision
and the explanation shown to the reviewer can never disagree.

## Mock customers

The customer book is chosen so every branch has something to exercise it.
`python -m src.graph` prints this table:

| Customer | TOI (VND) | Churn | Proposed action | Conf. | Route |
|---|---|---|---|---|---|
| CUST001 | 480,000,000 | 92% | `increase_credit_limit` | 0.91 | human — **hard rule beats high confidence** |
| CUST002 | 90,000,000 | 8% | `send_email` | 0.92 | auto-execute |
| CUST003 | 120,000,000 | 25% | `send_email` | 0.82 | human — below threshold |
| CUST004 | 650,000,000 | 72% | `increase_credit_limit` | 0.75 | human — hard rule *and* low confidence |
| CUST005 | 520,000,000 | 15% | `send_email` | 0.88 | auto-execute |
| CUST006 | 85,000,000 | 78% | `send_email` | 0.84 | human — 0.01 below threshold |

CUST001 is the interesting one for the demo: confidence 0.91 clears the 0.85
threshold, and it *still* goes to a human, because the hard rule is checked
first. CUST003 reproduces the lab's own example (`send_email` at 0.82).

Reasoning is deterministic — the same customer always yields the same proposal,
so what the lab demonstrates is the HITL behaviour rather than sampling noise.
Swapping `assess_customer()` for a real LLM call is the only change needed to
put a model behind it; nothing downstream depends on how the proposal was made.

## Demoing each verification case

| To check | Do this |
|---|---|
| Hard rule beats confidence | Run **CUST001** — pauses at 0.91 |
| Auto-execute | Run **CUST002** — completes with no review |
| Escalation below threshold | Run **CUST003** — pauses at 0.82 |
| State survives the pause | Expand *Customer record* on a paused card |
| Approve | Paused card → **Approve** → outcome shows `APPROVED … executed` |
| Reject | Paused card → **Reject** → outcome shows `REJECTED … Nothing changed` |
| Edit | Paused card → **Edit** → change 50,000,000 to 20,000,000 → submit |
| Audit trail | The table at the bottom, or `audit_log.json` |

## Design notes

**Why the executor node is also the gate.** `interrupt_before` pauses *ahead of*
`execute_high_risk_action`, so that node is what runs on resume. It therefore
dispatches on `human_decision` itself: approve and edit execute, reject returns
without any side effect. Both paths write an audit entry, so nothing reaches
`END` unlogged.

**Fail-closed default.** If `execute_high_risk_action` is ever reached with no
decision on record, it refuses to execute and logs `missing_decision`. A
high-risk action should never run because a decision went missing — the absence
of approval is not approval.

**`GraphState` has more than five keys.** The five the lab requires, plus
supporting keys (`customer`, `action_amount`, `route_reason`, `reviewer_id`,
`executed`, …). These matter because they must survive the interrupt too: the
reviewer needs the customer record and the proposed amount to make a decision,
and the resumed node needs the edited amount to execute.

**`AuditEntry` has more than six fields.** The six required ones come first,
then `customer_id`, `reasoning`, `route_reason`, `executed` and `details`. Six
fields alone tell you an action was approved but not which customer it touched.

**`reviewer_id` for auto-executed actions** is the sentinel `system:auto` rather
than empty, so "did a person review this?" stays a single-field query.

## Limitations

These are deliberate scope choices for a lab, and each is the wrong choice in
production:

- **`MemorySaver` is in-memory.** Paused runs live in the Streamlit process, so
  restarting the server loses them. Production needs a durable checkpointer
  (`SqliteSaver`, `PostgresSaver`) so a run can wait for a reviewer across a
  deploy.
- **The audit log is a JSON file rewritten on every append.** Read-append-write
  keeps history intact for a single process, but two concurrent reviewers can
  interleave and lose an entry. Production needs an append-only table.
- **Actions are simulated.** Executing an action formats a string; it does not
  call a core-banking API.
- **Confidence is not calibrated.** It is a deterministic function of the input
  features, not a measured probability of correctness — see question 3 in
  [REFLECTION.md](REFLECTION.md).
