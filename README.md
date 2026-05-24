# IDU -- Computational Demonstration

Reference implementation and demonstrations for the architecture in
*Interpretation, Learning, and Empathy as One Constraint: A Residual-Adequacy
Architecture with Accountable Abstention*.

This is the **linear instance** of the architecture: each regime reads content
through orthogonal projection onto a private subspace, and the residual is the
norm of what that projection fails to carry. The engine requires only a
well-defined interpretation map yielding a residual, an MDL-gated commit, and
deterministic finite-cost lookups (Theorem 1); linearity is the binding chosen
here. Everything is deterministic and reproducible to the bit.

## Files

- `idu.py` -- the engine: regimes, misfit `rho_i`, activation, residual `r(c)`,
  Regime--Act and Act-conflict graphs, the Decision loop (fixed priority
  HALT -> counter -> residual/action), focus-limited MDL basis expansion, the
  typed witnesses, and counterfactual simulation.
- `demo_abstention.py` -- one engine, three requests, the three witnessed
  terminals (`freeze_time`, `freeze_resid`, `freeze_halt`).
- `demo_empathy.py` -- two agents, shared label, private bases; agreement on
  clear-cut content, localized divergence on ambiguous content.
- `demo_prerequisites.py` -- a target that is `stuck` until a prerequisite basis
  concentrates its residual within the focus window `w`; agent-relative.
- `make_figure.py` -- regenerates the three-panel figure `demo_figure.png`.
- `run_all.py` -- runs all three demos and the figure.
- `idu_demo.ipynb` -- the same demonstrations as an inspectable notebook.

## Run

```
python run_all.py        # all three demos + figure
python demo_abstention.py
python demo_empathy.py
python demo_prerequisites.py
```

Requires only `numpy` and `matplotlib`.

## What each demonstration shows

| Demo | Phenomenon | Result |
|------|------------|--------|
| abstention | heterogeneity of not-knowing | three structurally distinct witnessed terminals from one engine |
| empathy | bounded mutual understanding | forced, self-invisible, localized divergence between shared-label/private-basis agents |
| prerequisites | learnability order | prerequisite induced by residual concentration within `w`, agent-relative to focus capacity |
