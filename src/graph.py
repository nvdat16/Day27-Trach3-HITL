"""LangGraph workflow for churn-risk assessment with Human-in-the-Loop control.

    customer_id
         |
         v
  evaluate_customer        -> proposed_action, confidence_score, reasoning
         |
         v
    route_action           -> hard rule first, confidence second
         |
    +----+---------------------------+
    | low risk                      | high risk / needs review
    v                               v
execute_low_risk_action     [ INTERRUPT BEFORE ]
    |                       execute_high_risk_action
    |                               |
    +---------------+---------------+
                    v
                  audit -> END
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

try:
    from src.models import (
        AGENT_ID,
        SYSTEM_REVIEWER,
        AuditEntry,
        append_audit_entry,
        now_iso,
    )
except ModuleNotFoundError:
    from models import (
        AGENT_ID,
        SYSTEM_REVIEWER,
        AuditEntry,
        append_audit_entry,
        now_iso,
    )

# --------------------------------------------------------------------------
# Policy constants
# --------------------------------------------------------------------------

#: Confidence at or above this may auto-execute -- but only a low-risk action.
CONFIDENCE_THRESHOLD = 0.85

#: Hard policy rules. These actions always go to a human, whatever the score.
HIGH_RISK_ACTIONS = {"increase_credit_limit"}

#: Total Operating Income above which a customer counts as high value.
HIGH_VALUE_TOI_VND = 300_000_000

#: Churn probability at or above which the customer is treated as at risk.
CHURN_ALERT_PROBABILITY = 0.70

LOW_RISK_NODE = "execute_low_risk_action"
HIGH_RISK_NODE = "execute_high_risk_action"


# --------------------------------------------------------------------------
# Step 1 - Graph state
# --------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    """State carried across the whole workflow, including across the pause.

    ``total=False`` because the graph starts with only ``customer_id`` set and
    fills the rest in as it goes.
    """

    # --- required by the lab spec ---
    customer_id: str
    proposed_action: str
    confidence_score: float
    reasoning: str
    human_decision: str | None

    # --- supporting keys; these must survive the interrupt too ---
    customer: dict          # raw customer record the assessment was based on
    action_amount: int      # VND, meaningful for increase_credit_limit
    route_reason: str       # policy_override | auto_execute | low_confidence
    reviewer_id: str
    review_note: str
    executed: bool
    execution_result: str
    audit_entry: dict


# --------------------------------------------------------------------------
# Mock customer book
#
# Each record is chosen to land on a different branch of the routing logic, so
# the verification steps in the lab all have a customer to exercise them.
# --------------------------------------------------------------------------

CUSTOMERS: dict[str, dict] = {
    "CUST001": {
        "customer_id": "CUST001",
        "name": "Nguyen Van An",
        "segment": "Priority",
        "toi_vnd": 480_000_000,
        "churn_probability": 0.92,
        "tenure_months": 54,
        "support_tickets_90d": 3,
        "competitor_offer": True,
    },
    "CUST002": {
        "customer_id": "CUST002",
        "name": "Tran Thi Binh",
        "segment": "Mass",
        "toi_vnd": 90_000_000,
        "churn_probability": 0.08,
        "tenure_months": 31,
        "support_tickets_90d": 1,
        "competitor_offer": False,
    },
    "CUST003": {
        "customer_id": "CUST003",
        "name": "Le Minh Cuong",
        "segment": "Mass",
        "toi_vnd": 120_000_000,
        "churn_probability": 0.25,
        "tenure_months": 12,
        "support_tickets_90d": 2,
        "competitor_offer": False,
    },
    "CUST004": {
        "customer_id": "CUST004",
        "name": "Pham Thu Dung",
        "segment": "Priority",
        "toi_vnd": 650_000_000,
        "churn_probability": 0.72,
        "tenure_months": 8,
        "support_tickets_90d": 6,
        "competitor_offer": True,
    },
    "CUST005": {
        "customer_id": "CUST005",
        "name": "Hoang Gia Em",
        "segment": "Priority",
        "toi_vnd": 520_000_000,
        "churn_probability": 0.15,
        "tenure_months": 76,
        "support_tickets_90d": 0,
        "competitor_offer": False,
    },
    "CUST006": {
        "customer_id": "CUST006",
        "name": "Vo Quoc Phong",
        "segment": "Mass",
        "toi_vnd": 85_000_000,
        "churn_probability": 0.78,
        "tenure_months": 5,
        "support_tickets_90d": 4,
        "competitor_offer": True,
    },
}


# --------------------------------------------------------------------------
# Step 2 - Agent reasoning
# --------------------------------------------------------------------------


def _suggested_limit_increase(toi_vnd: int) -> int:
    """10% of Total Operating Income, rounded to the nearest 5 million VND."""
    return int(round(toi_vnd * 0.10 / 5_000_000) * 5_000_000)


def assess_customer(customer: dict) -> tuple[str, float, str, int]:
    """Deterministic stand-in for an LLM churn assessment.

    Returns ``(proposed_action, confidence_score, reasoning, action_amount)``.

    Deterministic on purpose: the same customer always produces the same
    proposal, so the HITL behaviour is what the lab demonstrates rather than
    model sampling noise. Swap this one function for a real LLM call and
    nothing downstream changes.
    """
    churn = customer["churn_probability"]
    toi = customer["toi_vnd"]
    tickets = customer["support_tickets_90d"]

    at_risk = churn >= CHURN_ALERT_PROBABILITY
    high_value = toi >= HIGH_VALUE_TOI_VND

    # How decisive the churn estimate is: 0.5 is a coin flip, 0.0 or 1.0 is
    # clear-cut. Confidence tracks decisiveness, not churn risk itself.
    decisiveness = abs(churn - 0.5) * 2

    if at_risk and high_value:
        action = "increase_credit_limit"
        amount = _suggested_limit_increase(toi)
        confidence = 0.70 + 0.25 * decisiveness
        reasoning = (
            f"Churn probability {churn:.0%} is above the {CHURN_ALERT_PROBABILITY:.0%} "
            f"alert level and TOI of {toi:,.0f} VND puts this customer in the "
            f"high-value band. Raising the credit limit by {amount:,.0f} VND is "
            f"proposed as a retention lever."
        )
    else:
        action = "send_email"
        amount = 0
        confidence = 0.68 + 0.28 * decisiveness
        if at_risk:
            reasoning = (
                f"Churn probability {churn:.0%} is elevated, but TOI of {toi:,.0f} VND "
                f"is below the {HIGH_VALUE_TOI_VND:,.0f} VND high-value band, so a "
                f"financial concession is not justified. A retention email is proposed."
            )
        else:
            reasoning = (
                f"Churn probability {churn:.0%} is below the "
                f"{CHURN_ALERT_PROBABILITY:.0%} alert level and TOI is "
                f"{toi:,.0f} VND. No high-risk financial action is required; a "
                f"routine retention email is proposed."
            )

    # A noisy support history means the churn estimate rests on messier
    # evidence, so trust the assessment less.
    if tickets >= 5:
        confidence -= 0.06
        reasoning += f" Confidence reduced: {tickets} support tickets in the last 90 days."

    if customer.get("competitor_offer"):
        reasoning += " Customer has an open competitor offer on file."

    confidence = round(min(max(confidence, 0.0), 1.0), 2)
    return action, confidence, reasoning, amount


def evaluate_customer(state: GraphState) -> dict:
    """Agent node: assess the customer and propose an action with a confidence."""
    customer_id = state["customer_id"]
    customer = CUSTOMERS.get(customer_id)
    if customer is None:
        raise KeyError(
            f"Unknown customer_id {customer_id!r}. Known: {sorted(CUSTOMERS)}"
        )

    action, confidence, reasoning, amount = assess_customer(customer)
    _, route_reason = classify_route(action, confidence)

    return {
        "customer": customer,
        "proposed_action": action,
        "confidence_score": confidence,
        "reasoning": reasoning,
        "action_amount": amount,
        "route_reason": route_reason,
        # Every run starts with no human decision on record.
        "human_decision": None,
        "executed": False,
    }


# --------------------------------------------------------------------------
# Step 3 - Confidence routing and hard rules
# --------------------------------------------------------------------------


def classify_route(action: str, confidence: float) -> tuple[str, str]:
    """Resolve the routing decision to ``(next_node, route_reason)``.

    Shared by ``evaluate_customer`` (which stores the reason in state so the
    reviewer can see *why* an item reached them) and by ``route_action`` (which
    returns the node name). One function, so the two can never disagree.
    """
    # Rule 1 - policy override. Checked FIRST, before confidence is even read,
    # so a high score can never buy a bypass around the hard rule.
    if action in HIGH_RISK_ACTIONS:
        return HIGH_RISK_NODE, "policy_override"

    # Rule 2 - auto-execute: low-risk action the agent is confident about.
    if confidence >= CONFIDENCE_THRESHOLD:
        return LOW_RISK_NODE, "auto_execute"

    # Rule 3 - escalate: low risk, but not confident enough to act alone.
    return HIGH_RISK_NODE, "low_confidence"


def route_action(state: GraphState) -> str:
    """Conditional edge: pick the next node from the agent's output."""
    next_node, _ = classify_route(state["proposed_action"], state["confidence_score"])
    return next_node


# --------------------------------------------------------------------------
# Step 6 - Action execution and audit logging
# --------------------------------------------------------------------------


def _record_audit(
    state: GraphState,
    *,
    decision: str,
    reviewer_id: str,
    executed: bool,
    details: str,
) -> dict:
    """Build an AuditEntry, append it to audit_log.json, return it as a dict."""
    entry = AuditEntry(
        timestamp=now_iso(),
        agent_id=AGENT_ID,
        action=state["proposed_action"],
        confidence=state["confidence_score"],
        reviewer_id=reviewer_id,
        decision=decision,
        customer_id=state.get("customer_id", ""),
        reasoning=state.get("reasoning", ""),
        route_reason=state.get("route_reason", ""),
        executed=executed,
        details=details,
    )
    append_audit_entry(entry)
    return entry.model_dump()


def _describe_action(action: str, amount: int) -> str:
    if action == "increase_credit_limit":
        return f"increase_credit_limit by {amount:,.0f} VND"
    return action


def execute_low_risk_action(state: GraphState) -> dict:
    """Auto-execute path: no hard rule applies and confidence cleared the bar.

    This node is deliberately *not* in ``interrupt_before`` -- it runs without
    a human, which is the whole point of having a threshold.
    """
    action = state["proposed_action"]
    result = (
        f"{action} executed automatically for {state['customer_id']} "
        f"(confidence {state['confidence_score']:.2f} >= {CONFIDENCE_THRESHOLD})."
    )

    audit_entry = _record_audit(
        state,
        decision="auto_approve",
        reviewer_id=SYSTEM_REVIEWER,
        executed=True,
        details=result,
    )
    return {
        "executed": True,
        "execution_result": result,
        "audit_entry": audit_entry,
    }


def execute_high_risk_action(state: GraphState) -> dict:
    """High-risk path. Runs only *after* a human releases the interrupt.

    Because ``interrupt_before`` pauses ahead of this node, by the time the
    body executes a reviewer has written ``human_decision`` into the
    checkpoint via ``graph.update_state()``. The node then dispatches on it:

    * approve -> execute the action as proposed
    * edit    -> execute the action the reviewer rewrote into state
    * reject  -> abort, no side effect at all
    """
    decision = (state.get("human_decision") or "").strip().lower()
    action = state["proposed_action"]
    amount = state.get("action_amount", 0)
    reviewer_id = state.get("reviewer_id") or "unknown"
    note = state.get("review_note") or ""

    if decision == "approve":
        executed = True
        result = (
            f"APPROVED by {reviewer_id}: {_describe_action(action, amount)} "
            f"executed for {state['customer_id']}."
        )
    elif decision == "edit":
        executed = True
        result = (
            f"EDITED by {reviewer_id}: {_describe_action(action, amount)} "
            f"executed for {state['customer_id']} using the reviewer's values."
        )
    elif decision == "reject":
        executed = False
        result = (
            f"REJECTED by {reviewer_id}: {_describe_action(action, amount)} "
            f"was not executed for {state['customer_id']}. Nothing changed."
        )
    else:
        # Fail closed. Reaching this node without a decision means the
        # interrupt was bypassed, and a high-risk action must never execute on
        # a missing decision.
        decision = "missing_decision"
        executed = False
        result = (
            f"ABORTED: no human decision on record for {state['customer_id']}; "
            f"refusing to execute {action}."
        )

    if note:
        result += f" Reviewer note: {note}"

    audit_entry = _record_audit(
        state,
        decision=decision,
        reviewer_id=reviewer_id,
        executed=executed,
        details=result,
    )
    return {
        "executed": executed,
        "execution_result": result,
        "human_decision": decision,
        "audit_entry": audit_entry,
    }


# --------------------------------------------------------------------------
# Step 4 - Compile the graph with an interrupt
# --------------------------------------------------------------------------


def build_graph():
    """Compile the workflow with a checkpointer and a pause before high risk.

    ``MemorySaver`` is what lets the run survive the pause: the customer data
    and the agent's proposal stay in the checkpoint while the reviewer decides.
    Without a checkpointer there is nothing to resume into.
    """
    builder = StateGraph(GraphState)

    builder.add_node("evaluate_customer", evaluate_customer)
    builder.add_node(LOW_RISK_NODE, execute_low_risk_action)
    builder.add_node(HIGH_RISK_NODE, execute_high_risk_action)

    builder.add_edge(START, "evaluate_customer")
    builder.add_conditional_edges(
        "evaluate_customer",
        route_action,
        {
            LOW_RISK_NODE: LOW_RISK_NODE,
            HIGH_RISK_NODE: HIGH_RISK_NODE,
        },
    )
    builder.add_edge(LOW_RISK_NODE, END)
    builder.add_edge(HIGH_RISK_NODE, END)

    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=[HIGH_RISK_NODE],
    )


def thread_config(thread_id: str) -> dict:
    """Config for one workflow run.

    The same ``thread_id`` must be used to invoke, inspect and resume a run --
    that is how the checkpointer finds the paused state again.
    """
    return {"configurable": {"thread_id": thread_id}}


if __name__ == "__main__":
    # Quick smoke run: show where each mock customer gets routed.
    graph = build_graph()
    print(f"{'customer':<10} {'action':<22} {'conf':>5}  route")
    print("-" * 62)
    for cid in CUSTOMERS:
        config = thread_config(f"smoke-{cid}")
        state = graph.invoke({"customer_id": cid}, config=config)
        snapshot = graph.get_state(config)
        route = snapshot.next[0] if snapshot.next else "completed (auto-executed)"
        print(
            f"{cid:<10} {state['proposed_action']:<22} "
            f"{state['confidence_score']:>5.2f}  {route}"
        )
