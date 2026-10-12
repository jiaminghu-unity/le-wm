# Dynamics Closure Report — Tactile-Gated Residual Dynamics (2026-10-12)

Evidence base: `D0_FINAL_EVIDENCE.md` (frozen tables; protocol identity,
determinism and label alignment all verified there). This report answers the
five closure questions and issues the G1 decision.

## Q1 — T6 rankings, Top-K regret, and contact concentration

Six-step sequence diagnostic (80 held-out-sampled states × 100 demo-prior
sequences, executed in sim):

- Predicted cost vs physical outcome: Spearman **+0.47** (B1) / **+0.50** (DINO);
  oracle cost (encode the real final frame): **+0.64** / **+0.61**;
  pred-vs-oracle agreement 0.78/0.76.
- **Top-8 overlap** with the truly-best 8 sequences: pred **0.31**, oracle 0.42 —
  CEM's elite set built on predicted cost contains ~2.5/8 genuinely good
  sequences.
- Success-flag translation: where success is geometrically reachable (35/80
  states), the predicted-cost argmin reaches it **69%** vs oracle **89%** (B1).
  For DINO, oracle is no better than pred (74%/74%) — its cost geometry has no
  headroom at all.
- **Failures are NOT contact-concentrated**: pred Spearman near-transition
  +0.50 vs no-transition +0.44; contact +0.45 vs free +0.53 (oracle drops in
  parallel). One-step prediction with proper context is excellent (ρ +0.69,
  beats oracle).

**Conclusion:** multi-step degradation is generic autoregressive compounding
plus latent-cost slack — not a contact-dynamics deficit. The pre-registered
pass bar for continuing Dynamics ("a specific, measurable, action-selection-
relevant prediction defect in contact-related states") is **not met**.

## Q2 — Same Ours checkpoint, four inference paths

Held-out AR prediction MSE (1827 windows):

| route | H1 | H3 | H6 | H1 contact | H1 transition |
|---|---|---|---|---|---|
| Base (P0 only) | **0.00426** | 0.01705 | 0.04452 | **0.00578** | **0.00580** |
| Soft | 0.00441 | **0.01549** | **0.03885** | 0.00611 | 0.00644 |
| Hard (deployed) | 0.00452 | 0.01564 | 0.03904 | 0.00634 | 0.00680 |
| Oracle (true label) | 0.00469 | 0.01655 | 0.04081 | 0.00683 | 0.00689 |

Oracle routing is the **worst** at H1 and worse than learned routing at H3/H6 —
including on contact and transition subsets. Closed-loop: Ours-hard 15.8%,
Ours-soft 17.0% (n.s. apart), both significantly below B1 20.8%. Note: Ours'
P0 is the jointly-trained base inside the checkpoint, not the independent B1 —
"Base" is an internal ablation.

## Q3 — Are R1/R2 distinct, control-useful corrections?

They are distinct (cos(R1,R2)=0.09) and genuinely mode-specialized — **on the
training set**: cos(R1, Δ*)=0.72 on mode-1 transitions with optimal scale
α=1.05 (R2: 0.75/1.00 on mode-2), cutting the correction error ~65%. On
held-out episodes the same quantities collapse to cos ≈ 0.17–0.25, α ≈ 0.26–0.37,
and applying any residual (oracle included) **increases** error (mode-2:
0.0071→0.0085). Verdict: **pure generalization failure** — the per-mode
corrections are real but episode-specific; 141 training episodes cannot support
transferable contact corrections for a 512-hidden MLP. Not a direction error,
not a magnitude error, not a routing error.

B3's residuals are the mirror image: cos≈0.95 with Δ* on BOTH splits, identical
across modes, norms 5.4 (10× Ours) — they are not corrections but the generic
dynamics itself, absorbed into a soft mixture (gate degenerate: 95% mode-2,
macro-F1 0.17). B3-soft is functionally a single bigger predictor.

## Q4 — How much of the B3/Ours gap was routing inconsistency?

Essentially all of it. Offline: deploying B3 with argmax inflates its H1 error
×11 and H6 ×32 versus its soft training mode. Closed-loop, paired McNemar:

- B3 soft vs hard: **+6.0pp, p=0.0001** — the routing fix alone.
- Ours-soft vs B3-soft (fair comparison): **+0.5pp, p=0.87** — the tactile CE's
  independent closed-loop contribution is **zero**.
- The historical "tactile rescues multi-branch" (+5.3pp hard) was an artifact:
  CE + teacher forcing kept the dynamics inside P0 (small residuals), which
  incidentally immunized Ours against the deploy mismatch that crippled B3.

The gate itself did learn tactile modes (held-out macro-F1 0.61 vs B3's 0.17),
and routing-on-predicted-latents drifts 25–40% by step 5–6 in contact modes —
but none of this matters because mode-routed corrections carry no held-out value
(Q3) and oracle routing already loses to base (Q2).

## Q5 — Is D3 (single R vs multi R) worth compute?

**No.** The evidence already contains the answer at zero cost: B3-soft *is* the
single-effective-residual model (one soft mixture ≈ one bigger predictor). It is
the best offline one-step predictor of the family (H1 0.0012, 3.4× better than
any Ours route) and still loses closed-loop to the no-residual B1 by 4.2pp
(p=0.024). Enforced mode specialization (Ours) memorizes; un-enforced capacity
(B3-soft) helps offline MSE but not control. A D3 training run could only
re-measure this with worse controls.

## G1 — Decision

**Case B: STOP Tactile-Gated Residual Dynamics.** Both B3-soft (16.5%) and Ours
(15.8/17.0%) sit significantly below plain B1 (20.8%). No evidence supports
adding residual branches under the current task, data and planning protocol;
the tactile-defined discrete decomposition adds nothing once routing is fair.

### What survives this closure (established findings)

1. Contact modes are physically real (1.8× matched-command motion difference)
   and wrist-predictable (gate F1 0.61; probe F1 0.60–0.64).
2. True per-mode corrections exist and are learnable in-sample — the limiting
   factor is demo scale, not the hypothesis's physics.
3. Multi-view fusion is an asset (B1 20.8 > wrist 14.2 > front 8.5).
4. The deployed representations lose contact information they were fed
   (raw wrist probe F1 0.601 vs joint latent 0.527 / LeWM hidden 0.544; N0).
5. The gap to the official 29.3% is generic: multi-step compounding + latent
   cost slack (oracle top-1 regret 0.20) + 201-demo coverage — not contact
   dynamics, not the action-context bug (fixed), not routing.

### N-track status (annex)

N0 ran under the freeze (probes only, zero training): visual history adds **no**
mode-classification information (wrist single-frame 0.601 is the best row;
h3/h6/h12 flat-to-worse) and a **modest** motion-prediction gain at transitions
(R² 0.582→0.638). The temporal motive is weak; the stronger, already-replicated
motive remains representation retention. N1–N3 stay frozen pending a new
necessity case.

### Rule going forward (adopted)

Every proposed experiment must first answer: *which decision does its expected
result change?* Residual widths, gate temperatures and mode-count sweeps all
fail this test after Q2's oracle result.
