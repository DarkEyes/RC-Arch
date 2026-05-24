"""Generate the demonstration figure: three panels, one per instantiation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import importlib
emp = importlib.import_module("demo_empathy")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

# ---- Panel A: typed abstention -- three terminals -------------------------
ax = axes[0]
ax.set_title("(a) Typed abstention", fontsize=11, fontweight="bold")
terms = ["freeze_time", "freeze_resid", "freeze_halt"]
causes = ["conflict\n(advocacy vs summary)", "unrepresentable\n(coord 2)", "values stop\n(policy HALT)"]
colors = ["#d98c00", "#7a4fb3", "#c0392b"]
for i, (t, ca, col) in enumerate(zip(terms, causes, colors)):
    ax.add_patch(FancyBboxPatch((0.08, 0.72 - i*0.30), 0.84, 0.20,
                 boxstyle="round,pad=0.02", linewidth=1.4,
                 edgecolor=col, facecolor=col+"22"))
    ax.text(0.16, 0.82 - i*0.30, t, fontsize=10, fontweight="bold", color=col, va="center")
    ax.text(0.16, 0.76 - i*0.30, ca, fontsize=7.5, color="#333", va="center")
ax.text(0.5, 0.04, "one engine, three witnessed causes", fontsize=8,
        ha="center", style="italic", color="#555")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ---- Panel B: bounded empathy -- agreement vs divergence ------------------
ax = axes[1]
ax.set_title("(b) Bounded empathy", fontsize=11, fontweight="bold")
sweep = emp.RESULTS["sweep"]
xs = [s[1] for s in sweep]                 # psychosomatic fraction
agree = [1 if s[4] else 0 for s in sweep]
ax.fill_between(xs, 0, agree, step="mid", color="#2e8b57", alpha=0.25, label="actions agree")
ax.fill_between(xs, agree, 1, step="mid", color="#c0392b", alpha=0.25, label="actions diverge")
ax.plot(xs, agree, drawstyle="steps-mid", color="#2e8b57", linewidth=1.5)
ax.set_xlabel("content: psychosomatic fraction", fontsize=8.5)
ax.set_yticks([0, 1]); ax.set_yticklabels(["diverge", "agree"], fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
ax.text(0.5, 0.5, "ambiguous\nband", fontsize=8, ha="center", va="center", color="#c0392b")
ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
ax.text(0.5, -0.32, f"shared label, private bases ({emp.RESULTS['angle_deg']:.0f}\u00b0 apart)",
        transform=ax.transAxes, fontsize=8, ha="center", style="italic", color="#555")

# ---- Panel C: developmental prerequisites -- stuck -> learnable ------------
ax = axes[2]
ax.set_title("(c) Developmental prerequisites", fontsize=11, fontweight="bold")
states = ["before\n(only fn-notation)", "after\n(+algebra,+limits)", "wider focus\n(w=3)"]
unrep = [3, 1, 3]; learn = [False, True, True]; wbar = [1, 1, 3]
xpos = np.arange(3)
bars = ax.bar(xpos, unrep, width=0.55,
              color=["#c0392b" if not l else "#2e8b57" for l in learn], alpha=0.6)
for i, (u, l, wv) in enumerate(zip(unrep, learn, wbar)):
    ax.hlines(wv, i-0.32, i+0.32, color="#222", linestyle="--", linewidth=1.3)
    ax.text(i, u+0.12, "learnable" if l else "stuck", ha="center", fontsize=8,
            color="#2e8b57" if l else "#c0392b", fontweight="bold")
ax.text(2.55, 3, "w (focus)", fontsize=7, color="#222", va="center")
ax.set_xticks(xpos); ax.set_xticklabels(states, fontsize=7.5)
ax.set_ylabel("unrepresented residual coords", fontsize=8.5)
ax.set_ylim(0, 3.8)
ax.text(0.5, -0.34, "prerequisite = residual concentration within w",
        transform=ax.transAxes, fontsize=8, ha="center", style="italic", color="#555")

plt.tight_layout()
plt.savefig("demo_figure.png", dpi=200, bbox_inches="tight")
print("wrote demo_figure.png")
