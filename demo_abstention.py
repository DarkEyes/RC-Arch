"""
Demonstration 1 -- Typed abstention: ONE engine, three inputs, three terminals.

A single fixed configuration meets three different content vectors and produces
the three distinct witnessed terminals. With a presence floor phi on activation,
a regime applies only when its block carries live signal it can represent, so
the same engine reaches freeze_time, freeze_resid, or freeze_halt depending only
on the input -- not on a reconfiguration.
"""
import numpy as np
from idu import Regime, Config, decide
np.set_printoptions(precision=3, suppress=True)

# Coordinates: 0,1 task block ; 2 evidence ; 3 policy.
n = 4
# All regimes use presence floor phi=0.5: active only when their block is live.
R_sum = Regime("summarize", coords=[0, 1], basis=np.eye(2), theta=0.15, n=n, phi=0.5)
R_adv = Regime("advocate",  coords=[0, 1], basis=np.eye(2), theta=0.15, n=n, phi=0.5)
R_pol = Regime("policy",    coords=[3],    basis=np.eye(1), theta=0.15, n=n, phi=0.5)
# Note: no regime owns coordinate 2 -> any signal there is unrepresentable.

G = {("summarize",): {"emit_summary"},
     ("advocate",):  {"emit_advocacy"},
     ("policy",):    {"HALT"}}
G_cf = {frozenset({"emit_summary", "emit_advocacy"})}

cfg = Config(regimes=[R_sum, R_adv, R_pol], G=G, G_cf=G_cf,
             theta_r=0.20, tau=0.30, n=n, w=2, t_max=8)

print("=" * 70); print("DEMONSTRATION 1 -- TYPED ABSTENTION (one engine, three inputs)"); print("=" * 70)

cases = [
    ("i  conflict",        np.array([1.0, 1.0, 0.0, 0.0])),  # task live -> sum+adv conflict
    ("ii unrepresentable", np.array([0.0, 0.0, 5.0, 0.0])),  # signal on coord 2 (no owner)
    ("iii values stop",    np.array([0.0, 0.0, 0.0, 1.0])),  # policy live -> HALT
]
results = {}
for name, c in cases:
    on = [r.label for r in cfg.regime_on(c)]
    w = decide(cfg, c)
    print(f"\n({name})  c={c}")
    print(f"   active regimes = {on}")
    print(f"   -> {w}")
    results[name.split()[0]] = w

print("\n" + "-" * 70)
print("One configuration, three contents, three witnessed terminal TYPES.")
RESULTS = results
