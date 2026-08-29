# Reflection Questions — Lab 27

## Question 1 — `interrupt_before` or `interrupt_after`?

> If the goal is to let a human rewrite a generated retention email before it
> moves to a routing node, would you use `interrupt_before` or `interrupt_after`?

**`interrupt_after=["draft_email"]`** — pause *after* the node that generates the
email, not before the routing node that consumes it.

The reason is simply that you cannot edit something that does not exist yet.
`interrupt_before=["draft_email"]` pauses while the state still has no draft in
it, so the reviewer is handed a blank card. The draft has to be produced and
written into state first; only then is there text to rewrite.

The general rule this illustrates:

- **`interrupt_before`** guards a **side effect** you must not perform without
  approval. That is Step 4 of this lab: `increase_credit_limit` changes a
  customer's credit line, so the pause has to land *before* the node that does
  it. Pausing after would mean approving something already done.
- **`interrupt_after`** exposes an **artifact** you must inspect before something
  else consumes it. A generated email is an artifact — producing it costs
  nothing and is fully reversible, and what needs controlling is not the
  generation but what happens next.

So the question to ask is not "where in the sequence is the human?" but "is the
thing I am guarding an *effect* or a *product*?" Effects get `interrupt_before`;
products get `interrupt_after`.

One practical note: in a simple linear graph, `interrupt_after=["draft_email"]`
and `interrupt_before=["route_email"]` pause at the same moment, so either
appears to work. They stop being equivalent as soon as the graph grows. If a
second edge later also points at `route_email`, `interrupt_before=["route_email"]`
starts pausing runs that never drafted an email; if the routing node is renamed
or split in two, the pause point silently drifts away from the thing being
reviewed. Anchoring the interrupt to the node that *produces* the artifact keeps
the review attached to its subject through refactors.

---

## Question 2 — Preventing alert fatigue

> The UI forces a human to review 500 `send_email` actions per day because
> confidence sits at 0.82, just below the 0.85 threshold. What would you change?

The queue is a symptom. The root cause is that a **single global threshold is
being asked to answer two unrelated questions**: *how sure is the agent?* and
*how much does a mistake cost?* Confidence answers the first. The threshold is
being used to decide the second, which it cannot do. A wrong retention email
costs a mildly annoyed customer; a wrong credit-limit increase costs real money
and is hard to claw back. Gating both at 0.85 is what manufactures 500 reviews
that nobody reads carefully by item 40 — and a reviewer who rubber-stamps 500
items is providing *less* safety than no review at all, because the system now
carries an audit trail that falsely claims human oversight.

### The main fix: gate on reversibility × cost, not confidence alone

Replace the single threshold with a per-action policy:

| Action | Reversible? | Cost of error | Gate |
|---|---|---|---|
| `send_email` | yes, an apology undoes most of it | low | hard rules only; no confidence gate, or ~0.60 |
| `offer_voucher` | partly | bounded by voucher value | threshold scaled to value |
| `increase_credit_limit` | no | high | hard rule — always human |

This alone removes essentially all 500 reviews while *strengthening* the control
on the actions that matter, because the reviewer's attention is no longer spent
on emails.

### Supporting changes

1. **Check whether 0.85 is even the right number.** A cluster stuck at 0.82 is a
   symptom of a compressed, uncalibrated score. Measure the actual accuracy of
   the 0.82 bucket in the audit log. If those emails are right 97% of the time,
   the threshold is wrong, not the agent — and lowering it is a
   measurement-backed decision rather than a capitulation.
2. **Move reversible actions from pre-approval to post-hoc audit.** Auto-execute
   them and route a sample into a review queue that does not block. The reviewer
   is still in the loop; they are just no longer a bottleneck on a reversible
   action.
3. **Sample instead of enumerating.** Review 5% at random with a
   circuit-breaker: if the sampled error rate exceeds a bound, full review turns
   back on automatically. Oversight stays continuous, effort drops ~20×.
4. **Cluster, because 500 near-identical actions are one decision.** Group by
   template plus customer segment and approve per cluster with a
   drill-down. Ten clusters of fifty is a decision a human can actually make.
5. **Sort the queue by expected cost of error**, not arrival time, so that when
   attention runs out it runs out on the least consequential items.
6. **Make the fast path fast** — keyboard shortcuts, bulk select, no modal per
   item. This is the smallest-value change on the list, and it is where the
   instinct to "fix the UI" usually stops. It treats the symptom.

---

## Question 3 — Why not trust self-reported confidence?

> The agent reports 0.95 when proposing `increase_credit_limit` but is often
> wrong about the customer's actual income. Why is relying on the LLM's
> self-reported confidence dangerous, and how would you calibrate it before
> routing?

### Why it is dangerous

**The confidence and the error come from the same place.** Self-reported
confidence is produced by the same forward pass that produced the possibly wrong
answer, conditioned on the same misunderstanding. It is therefore not an
independent check — its mistakes are *correlated* with the answer's mistakes.
When the model has confidently misread the income, it is confidently sure about
its confidence too. A useful check has to be able to fail independently of the
thing it checks; this one cannot.

**It measures fluency, not correctness.** A high score means the completion
*reads* as authoritative given the prompt. Nothing in that signal is connected
to whether the income figure was retrieved from the right field.

**The specific failure here is a grounding failure, not a reasoning failure.**
The model doesn't know the customer's income; it read a field or invented one.
Confidence in a *conclusion* says nothing about whether the *input* was correct.
"Given an income of 480M, raising the limit is sensible" can be perfectly sound
reasoning and still be a bad action, if the income is actually 48M. Self-reported
confidence cannot distinguish these, because from the inside they are identical.

**Worst of all, routing turns the score into an authorisation token.** Once
`confidence >= 0.85` grants the right to act without a human, an
uncalibrated number has become a security boundary — and one the model gets to
set for itself. It is also silently unstable: a prompt tweak or a model upgrade
shifts the distribution, `0.85` quietly means something different, and no alarm
fires because nothing crashed. This lab's hard rule is exactly the mitigation:
confidence is allowed to *escalate*, never to *authorise*.

### How to calibrate before routing

**1. Verify the facts, don't grade the narrative** (highest value here, and
cheap). Add a verification step between reasoning and routing that checks the
agent's cited figures against the system of record. If the agent's claimed income
does not match the database, force escalation regardless of confidence. This
catches the exact described failure deterministically — no calibration theory
required.

**2. Cap confidence by input quality.**

```python
confidence = min(agent_confidence, data_quality_score)
```

Stale or missing income data caps the score low, so the item escalates on its
own. This encodes the rule that you cannot be more certain than your worst input.

**3. Fit a real calibrator on the audit log.** The audit trail is already the
training set: features (claimed confidence, churn probability, data
completeness, segment) against the outcome (did the reviewer approve, and was it
right). Isotonic regression or Platt scaling maps those to an empirical
P(correct). Then route on the calibrated probability, not the self-report.

**4. Measure calibration explicitly before trusting any threshold.** Bucket
historical predictions by claimed confidence, compute the observed accuracy per
bucket, and plot a reliability diagram with its Expected Calibration Error. If
the 0.95 bucket is right 62% of the time, that is the number the router should
see. This also makes the periodic re-check concrete rather than aspirational.

**5. Get an independent uncertainty signal.** Self-consistency: sample the
assessment k times and use the agreement rate. Disagreement across samples is
genuine uncertainty, and — unlike self-report — it is not generated by the same
single pass as the answer. Token log-probabilities on the decisive token are a
weaker but cheaper independent measurement.

**6. Keep the hard rules regardless.** Calibration makes the routing decision
better; it does not make it safe. `increase_credit_limit` should require a human
even at a perfectly calibrated 0.99, because the cost of being wrong is
asymmetric and the action is hard to reverse. Calibration and policy solve
different problems, and the second one is not optional.
