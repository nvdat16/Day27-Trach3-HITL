"""Streamlit approval console for the HITL churn-risk workflow.

Run with:

    streamlit run src/app.py

The console does three things: start a workflow run, show the action card for a
run that LangGraph has paused, and resume that run with the reviewer's decision.
"""

from __future__ import annotations

from uuid import uuid4

import streamlit as st

try:
    from src.graph import (
        CONFIDENCE_THRESHOLD,
        CUSTOMERS,
        HIGH_RISK_ACTIONS,
        HIGH_RISK_NODE,
        build_graph,
        thread_config,
    )
    from src.models import AGENT_ID, load_audit_log
except ModuleNotFoundError:
    from graph import (
        CONFIDENCE_THRESHOLD,
        CUSTOMERS,
        HIGH_RISK_ACTIONS,
        HIGH_RISK_NODE,
        build_graph,
        thread_config,
    )
    from models import AGENT_ID, load_audit_log

st.set_page_config(
    page_title="Churn Risk HITL Console",
    page_icon="🛡️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Session state
#
# The compiled graph holds the MemorySaver checkpointer, so it must be built
# once and kept: rebuilding it on every Streamlit rerun would throw away the
# paused run we are trying to review.
# --------------------------------------------------------------------------

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
st.session_state.setdefault("thread_id", None)
st.session_state.setdefault("show_edit", False)

graph = st.session_state.graph


ROUTE_EXPLANATION = {
    "policy_override": (
        "Hard policy rule. `{action}` always requires human review, whatever the "
        "confidence score says ({confidence:.2f}). Policy is checked before the "
        "threshold, so a high score cannot buy a bypass."
    ),
    "low_confidence": (
        "Confidence {confidence:.2f} is below the {threshold:.2f} threshold. The "
        "action is low-risk, but the agent is not sure enough to act alone."
    ),
    "auto_execute": (
        "Confidence {confidence:.2f} cleared the {threshold:.2f} threshold on a "
        "low-risk action, so it ran without human review."
    ),
}


def customer_option_label(customer_id: str) -> str:
    return f"{customer_id} — {CUSTOMERS[customer_id]['name']}"


def customer_id_from_label(label: str) -> str:
    return label.split(" — ", 1)[0]


def explain_route(values: dict) -> str:
    template = ROUTE_EXPLANATION.get(values.get("route_reason", ""), "")
    return template.format(
        action=values.get("proposed_action", ""),
        confidence=values.get("confidence_score", 0.0),
        threshold=CONFIDENCE_THRESHOLD,
    )


def resume_graph(decision: str, reviewer_id: str, extra: dict | None = None) -> None:
    """Write the human decision into the checkpoint, then resume the run.

    ``update_state`` edits the paused state in place; ``invoke(None, config)``
    is what tells LangGraph to continue from that checkpoint instead of
    starting a fresh run.
    """
    config = thread_config(st.session_state.thread_id)
    updates = {"human_decision": decision, "reviewer_id": reviewer_id}
    if extra:
        updates.update(extra)

    graph.update_state(config, updates)
    graph.invoke(None, config=config)

    st.session_state.show_edit = False
    st.rerun()


# --------------------------------------------------------------------------
# Sidebar - start a run
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Reviewer")
    reviewer_id = st.text_input("Reviewer ID", value="operator_01")

    st.header("Customer")
    customer_label = st.selectbox(
        "Select a customer",
        options=[customer_option_label(cid) for cid in CUSTOMERS],
    )
    customer_id = customer_id_from_label(customer_label)

    picked = CUSTOMERS[customer_id]
    st.caption(
        f"TOI {picked['toi_vnd']:,.0f} VND · churn {picked['churn_probability']:.0%} · "
        f"{picked['support_tickets_90d']} tickets/90d"
    )

    if st.button(
        "Run assessment",
        type="primary",
        use_container_width=True,
        key="btn_run",
    ):
        st.session_state.thread_id = f"{customer_id}-{uuid4().hex[:8]}"
        st.session_state.show_edit = False
        graph.invoke(
            {"customer_id": customer_id},
            config=thread_config(st.session_state.thread_id),
        )

    st.divider()
    st.caption(f"Agent: `{AGENT_ID}`")
    st.caption(f"Auto-execute threshold: `{CONFIDENCE_THRESHOLD}`")
    st.caption("Hard rules: " + ", ".join(f"`{a}`" for a in sorted(HIGH_RISK_ACTIONS)))


# --------------------------------------------------------------------------
# Main - action card and review controls
# --------------------------------------------------------------------------

st.title("🛡️ Churn Risk — Human-in-the-Loop Console")

if st.session_state.thread_id is None:
    st.info("Pick a customer in the sidebar and select **Run assessment** to begin.")
    st.stop()

config = thread_config(st.session_state.thread_id)
snapshot = graph.get_state(config)
values = snapshot.values

# A non-empty `next` tuple *is* the pending signal — no separate flag to keep
# in sync with the graph.
pending = bool(snapshot.next)
awaiting_review = pending and snapshot.next[0] == HIGH_RISK_NODE

st.caption(f"Thread `{st.session_state.thread_id}`")

if awaiting_review:
    st.warning("⏸️ **Workflow paused.** The action below has not been executed yet.")
else:
    st.success("✅ **Workflow complete.**")

# --- Action card ---------------------------------------------------------

st.subheader("Action card")

col1, col2, col3 = st.columns([1, 2, 1])
col1.metric("Customer ID", values.get("customer_id", "—"))
col2.metric("Proposed action", values.get("proposed_action", "—"))
col3.metric("Confidence", f"{values.get('confidence_score', 0.0):.2f}")

st.progress(min(max(values.get("confidence_score", 0.0), 0.0), 1.0))

if values.get("action_amount"):
    st.metric("Proposed amount", f"{values['action_amount']:,.0f} VND")

st.markdown("**Reasoning**")
st.info(values.get("reasoning", "—"))

explanation = explain_route(values)
if explanation:
    st.markdown("**Why this route**")
    if awaiting_review:
        st.warning(explanation)
    else:
        st.success(explanation)

with st.expander("Customer record used for this assessment"):
    st.json(values.get("customer", {}))

# --- Review controls -----------------------------------------------------

if awaiting_review:
    st.divider()
    st.subheader("Your decision")

    note = st.text_input("Reviewer note (optional)", key="review_note_input")

    approve_col, reject_col, edit_col = st.columns(3)

    if approve_col.button("✅ Approve", use_container_width=True, key="btn_approve"):
        resume_graph("approve", reviewer_id, {"review_note": note})

    if reject_col.button(
        "⛔ Reject",
        use_container_width=True,
        type="secondary",
        key="btn_reject",
    ):
        resume_graph("reject", reviewer_id, {"review_note": note})

    if edit_col.button("✏️ Edit", use_container_width=True, key="btn_edit"):
        st.session_state.show_edit = not st.session_state.show_edit

    if st.session_state.show_edit:
        with st.form("edit_action"):
            st.markdown("**Edit the action before it executes**")

            action_options = ["increase_credit_limit", "send_email"]
            current_action = values.get("proposed_action", "send_email")
            new_action = st.selectbox(
                "Action",
                options=action_options,
                index=action_options.index(current_action)
                if current_action in action_options
                else 0,
            )
            new_amount = st.number_input(
                "Amount (VND)",
                min_value=0,
                step=5_000_000,
                value=int(values.get("action_amount", 0)),
                help="Only applies to increase_credit_limit.",
            )
            edit_note = st.text_input(
                "Reason for the edit",
                value="Amount reduced by reviewer.",
            )

            if st.form_submit_button("Save and execute edited action", type="primary"):
                resume_graph(
                    "edit",
                    reviewer_id,
                    {
                        "proposed_action": new_action,
                        "action_amount": int(new_amount),
                        "review_note": edit_note,
                    },
                )

# --- Outcome -------------------------------------------------------------

else:
    st.divider()
    st.subheader("Outcome")
    if values.get("executed"):
        st.success(values.get("execution_result", "Action executed."))
    else:
        st.error(values.get("execution_result", "Action was not executed."))

    decision = values.get("human_decision")
    st.caption(
        f"Recorded decision: `{decision or 'auto_approve (no human review required)'}`"
    )

# --------------------------------------------------------------------------
# Audit trail
# --------------------------------------------------------------------------

st.divider()
st.subheader("Audit trail")

entries = load_audit_log()
if not entries:
    st.caption("No audit entries yet.")
else:
    st.caption(f"{len(entries)} entries in `audit_log.json` (newest first).")
    st.dataframe(
        [
            {
                "timestamp": e.get("timestamp"),
                "customer": e.get("customer_id"),
                "action": e.get("action"),
                "confidence": e.get("confidence"),
                "reviewer": e.get("reviewer_id"),
                "decision": e.get("decision"),
                "executed": e.get("executed"),
            }
            for e in reversed(entries)
        ],
        use_container_width=True,
        hide_index=True,
    )
