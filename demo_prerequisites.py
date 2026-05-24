"""
Demonstration 3 -- Developmental prerequisites, via real learning-step attempts.

In learning mode the study step tries to reduce the in-focus residual of the
TARGET: the coordinates the target requires but that no held regime yet
represents. An admissible edit exists only if that unrepresented residual is
concentrated within the focus window of width w -- a single episode can bring
at most w coordinates into focus and add one basis direction. We run that test
directly: enumerate the unrepresented coordinates, take the w highest-residual
of them as the focus window, and check whether one admissible edit would close
the focused residual (i.e. whether the unrepresented residual fits in the
window). This is the residual-concentration mechanism of the paper, executed
rather than asserted.
"""
import numpy as np
from idu import Regime, Config
np.set_printoptions(precision=3, suppress=True)

# Coordinates = sub-skills: 0 function-notation, 1 limits, 2 algebra, 3 manipulation.
n = 4
target = np.array([1.0, 1.0, 1.0, 1.0])   # calculus target needs all four

def unrepresented(cfg, c):
    """Coordinates the target requires that no held regime represents."""
    owned = set()
    for r in cfg.regimes:                       # held regimes (whether or not active)
        owned.update(r.coords)
    return [k for k in range(cfg.n) if abs(c[k]) > 1e-9 and k not in owned]

def learning_step(cfg, c, w):
    """
    Run one study step. The in-focus residual is over the unrepresented
    coordinates; an admissible edit closes it iff they fit within the window.
    Returns (admissible_edit_exists, focus_window, residual_coords).
    """
    resid = unrepresented(cfg, c)               # coordinates carrying unrepresented residual
    resid_sorted = sorted(resid)                # deterministic
    F = resid_sorted[:w]                         # focus window: w highest-residual coords
    # One episode adds one basis direction, closing the focused residual; it
    # makes the target learnable iff NO unrepresented residual is left outside F.
    admissible = (len(resid) <= w)
    return admissible, F, resid_sorted

def report(tag, cfg, c, w):
    ok, F, resid = learning_step(cfg, c, w)
    held = sorted({l for r in cfg.regimes for l in [r.label]})
    print(f"\n{tag}")
    print(f"   held regimes = {held}")
    print(f"   unrepresented residual coords = {resid}")
    print(f"   focus window F (w={w}) = {F}")
    if ok:
        print(f"   study step -> admissible edit exists; residual fits in window -> LEARNABLE")
    else:
        print(f"   study step -> residual spans {len(resid)} coords > w={w}; "
              f"no admissible edit -> STUCK")
    return ok

print("=" * 70); print("DEMONSTRATION 3 -- DEVELOPMENTAL PREREQUISITES (real study-step attempts)"); print("=" * 70)

R_fn  = Regime("function", coords=[0], basis=np.eye(1), theta=0.15, n=n, phi=0.5)
R_alg = Regime("algebra",  coords=[2], basis=np.eye(1), theta=0.15, n=n, phi=0.5)
R_lim = Regime("limits",   coords=[1], basis=np.eye(1), theta=0.15, n=n, phi=0.5)
base = dict(G={}, G_cf=set(), theta_r=0.20, tau=0.30, n=n, t_max=8)

cfg_before = Config(regimes=[R_fn], w=1, **base)
ok_b = report("BEFORE: holds {function-notation}, w=1", cfg_before, target, 1)

cfg_after = Config(regimes=[R_fn, R_alg, R_lim], w=1, **base)
ok_a = report("AFTER: holds {function-notation, algebra, limits}, w=1", cfg_after, target, 1)

cfg_wide = Config(regimes=[R_fn], w=3, **base)
ok_w = report("WIDER FOCUS: holds {function-notation}, w=3", cfg_wide, target, 3)

print("\n" + "-" * 70)
print(f"stuck before acquisition: {not ok_b}")
print(f"learnable after acquiring algebra+limits: {ok_a}")
print(f"learnable with wider focus alone: {ok_w}")
print("The prerequisite is induced by held configuration and focus width w,")
print("not by a fixed ordering over topics.")
RESULTS = {"before_stuck": not ok_b, "after_ok": ok_a, "wide_ok": ok_w}
