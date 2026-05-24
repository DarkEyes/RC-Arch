"""
Interpretation--Decision Unit (IDU) -- reference implementation
================================================================

A faithful, deterministic implementation of the architecture specified in
"Interpretation, Learning, and Empathy as One Constraint: A Residual-Adequacy
Architecture with Accountable Abstention".

This is the *linear instance* of the architecture: each regime reads content
through an orthogonal projection onto a private subspace, and the residual is
the norm of what that projection fails to carry. The engine itself only
requires a well-defined interpretation map yielding a residual, an MDL-gated
commit, and deterministic finite-cost lookups (Theorem 1); linearity is the
binding chosen here.

Everything is deterministic and reproducible to the bit: no randomness, fixed
tie-breaking (lowest index / lexicographic), so repeated runs on the same
content and configuration yield the identical witness, as the totality theorem
requires.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# --------------------------------------------------------------------------
# Regime: a local representational frame (a "value space" in the sense of the
# cognitive-geometry companion paper). Linear instance: selector S_i + an
# orthonormal basis U_i whose column span is the subspace the regime can
# represent; P_i = U_i U_i^T is the orthogonal projector onto it.
# --------------------------------------------------------------------------

@dataclass
class Regime:
    label: str                 # ell_i : public label
    coords: list[int]          # D_i  : coordinate set (indices into R^n)
    basis: np.ndarray          # U_i  : orthonormal columns, shape (|D_i|, k_i)
    theta: float               # theta_i : activation tolerance on the misfit
    n: int                     # ambient dimension
    phi: float = 0.0           # phi_i : presence floor; block signal must reach it

    def __post_init__(self):
        self.coords = sorted(self.coords)
        if self.basis.size == 0:
            self.basis = np.zeros((len(self.coords), 0))
        # orthonormalize columns defensively (QR), keeping determinism
        if self.basis.shape[1] > 0:
            q, _ = np.linalg.qr(self.basis)
            self.basis = q[:, : self.basis.shape[1]]

    def _select(self, c: np.ndarray) -> np.ndarray:
        """S_i c restricted to the regime's coordinate block (length |D_i|)."""
        return c[self.coords]

    def _project_block(self, cb: np.ndarray) -> np.ndarray:
        """P_i acting within the coordinate block."""
        if self.basis.shape[1] == 0:
            return np.zeros_like(cb)
        return self.basis @ (self.basis.T @ cb)

    def misfit(self, c: np.ndarray) -> float:
        """rho_i(c) = || S_i c - P_i(S_i c) ||  -- the regime's misfit on c."""
        cb = self._select(c)
        return float(np.linalg.norm(cb - self._project_block(cb)))

    def presence(self, c: np.ndarray) -> float:
        """||S_i c|| : how much signal the regime's block carries."""
        return float(np.linalg.norm(self._select(c)))

    def is_active(self, c: np.ndarray) -> bool:
        """Active iff representable (rho_i(c) <= theta_i) AND present (||S_i c|| >= phi_i).

        The presence floor phi_i makes a regime apply only when its concept is
        live in the content: a silent block (||S_i c|| = 0) has zero misfit but
        carries no signal, so without a floor every regime would activate on
        silence. With phi_i > 0 a regime activates only when its block is both
        representable and actually present."""
        return (self.misfit(c) <= self.theta + 1e-12) and (self.presence(c) >= self.phi - 1e-12)

    def block_residual(self, c: np.ndarray) -> np.ndarray:
        """Residual vector within the block: (S_i c - P_i S_i c)."""
        cb = self._select(c)
        return cb - self._project_block(cb)

    def coord_residual(self, c: np.ndarray) -> dict[int, float]:
        """r_{i,k}(c) = |residual_k| for k in D_i, else 0."""
        res = self.block_residual(c)
        return {k: abs(float(res[j])) for j, k in enumerate(self.coords)}


# --------------------------------------------------------------------------
# Witness: every terminal returns W = (Regime_on, Act_on, X).
# --------------------------------------------------------------------------

@dataclass
class Witness:
    kind: str                       # 'emit' | 'freeze_halt' | 'freeze_time' | 'freeze_resid'
    regime_on: list[str]            # active labels
    act_on: list[str]               # licensed actions
    X: object = None                # unresolved structure (typed by kind)
    emitted: Optional[str] = None   # the emitted action, if kind == 'emit'

    def __str__(self):
        head = f"[{self.kind}]"
        body = f" regimes={self.regime_on} acts={self.act_on}"
        if self.kind == "emit":
            return f"{head} emit={self.emitted}{body}"
        return f"{head} X={self.X}{body}"


# --------------------------------------------------------------------------
# Configuration Gamma = ({R_i}, Actions, G, G_cf, {theta_i}, theta_r, tau)
# --------------------------------------------------------------------------

@dataclass
class Config:
    regimes: list[Regime]
    G: dict                         # regime-set (frozenset of labels) -> set of actions
    G_cf: set                       # set of frozenset({a, a'}) conflicting pairs
    theta_r: float                  # residual cutoff on r(c)
    tau: float                      # per-coordinate residual margin
    n: int                          # ambient dimension
    w: int = 2                      # focus-window size
    t_max: int = 16                 # re-entry bound
    HALT: str = "HALT"

    # ---- activation -----------------------------------------------------
    def regime_on(self, c):
        return [r for r in self.regimes if r.is_active(c)]

    def active_domain(self, c):
        on = self.regime_on(c)
        if not on:
            return list(range(self.n))      # empty-set convention
        D = set()
        for r in on:
            D.update(r.coords)
        return sorted(D)

    def coord_residual(self, c):
        """e_k(c) = max over active regimes of r_{i,k}(c)."""
        on = self.regime_on(c)
        D = self.active_domain(c)
        e = {k: 0.0 for k in D}
        if not on:                          # empty-set convention: r(c)=||c||
            for k in D:
                e[k] = abs(float(c[k]))
            return e
        for r in on:
            cr = r.coord_residual(c)
            for k, v in cr.items():
                e[k] = max(e[k], v)
        return e

    def residual(self, c):
        """r(c) = sum_k e_k(c)  (||c|| under the empty-set convention)."""
        e = self.coord_residual(c)
        if not self.regime_on(c):
            return float(np.linalg.norm(c))
        return float(sum(e.values()))

    # ---- action licensing ----------------------------------------------
    def act_on(self, c):
        """Act_on(c): actions licensed by Regime_on(c) through G."""
        on_labels = frozenset(r.label for r in self.regime_on(c))
        acts = set()
        for key, actions in self.G.items():
            if frozenset(key) <= on_labels:        # regime subset licenses its actions
                acts.update(actions)
        return sorted(acts)

    def conflicts(self, acts):
        """Pairs of licensed actions that conflict, via G_cf."""
        out = []
        for i in range(len(acts)):
            for j in range(i + 1, len(acts)):
                if frozenset({acts[i], acts[j]}) in self.G_cf:
                    out.append((acts[i], acts[j]))
        return out


# --------------------------------------------------------------------------
# Basis expansion: focus window -> candidate from residual -> MDL gate.
# Two-part code  L(data,M) = L_model(M) + L_resid(resid|M).
# Linear instance: candidate is the dominant residual direction in-focus.
# --------------------------------------------------------------------------

def focus_window(cfg: Config, c, e):
    """F(c): the w highest-residual coordinates (deterministic tie-break by index)."""
    D = cfg.active_domain(c)
    ranked = sorted(D, key=lambda k: (-e[k], k))
    return ranked[: cfg.w]

def mdl_gain(cfg: Config, c, F, lam=None):
    """
    Delta L = L(data, M) - L(data, M+v).
    L_model = lambda * (#dimensions). L_resid = sum of squared in-focus residuals.
    Adding one dimension v along the dominant in-focus residual absorbs that
    direction's energy; we credit the largest single-coordinate residual energy
    in focus as the data-cost saving (linear, single-direction instance).
    """
    e = cfg.coord_residual(c)
    if lam is None:
        lam = cfg.tau ** 2          # model cost of one dimension, in residual^2 units
    # data cost saved by absorbing the dominant in-focus residual coordinate
    saved = max((e[k] ** 2 for k in F), default=0.0)
    added = lam
    return saved - added            # > 0  <=> commit

def try_expand(cfg: Config, c):
    """
    Attempt one focus-limited, MDL-gated basis expansion.
    Returns (success, info). Success means a finite edit would be committed.

    Expansion can only extend a regime that ALREADY OWNS a focus coordinate
    (a candidate direction is added to such a regime's basis). If no active
    regime owns any focus coordinate, the residual is genuinely unrepresentable
    -- there is nothing to extend -- so expansion fails (-> freeze_resid). This
    is the structural difference between freeze_resid (no basis can absorb it)
    and a successful re-entry (a regime's basis grows to absorb it).
    """
    e = cfg.coord_residual(c)
    F = focus_window(cfg, c, e)
    on = cfg.regime_on(c)
    owned = set()
    for r in on:
        owned.update(r.coords)
    # focus coordinates that some active regime owns and could extend toward
    extendable = [k for k in F if k in owned]
    if not extendable:
        return False, {"focus": F, "reason": "no active regime owns focus coords"}
    dL = mdl_gain(cfg, c, extendable)
    return (dL > 0.0), {"focus": F, "extendable": extendable, "delta_L": dL}


# --------------------------------------------------------------------------
# Decision: fixed priority HALT -> counter -> residual/action branches.
# Returns the typed, witnessed terminal (Theorem 1).
# --------------------------------------------------------------------------

def decide(cfg: Config, c, verbose=False):
    t = 0
    last_reentry = None     # records the structure of the final re-entry (for freeze_time X)
    cur = c.copy()

    while True:
        on = cfg.regime_on(cur)
        on_labels = [r.label for r in on]
        acts = cfg.act_on(cur)

        # ---- HALT checked first ----
        if cfg.HALT in acts:
            halt_regimes = [r.label for r in on
                            if cfg.HALT in cfg.act_on_from_labels({r.label})]
            return Witness("freeze_halt", on_labels, acts,
                           X={"halt_regimes": halt_regimes or on_labels})

        # ---- counter bound ----
        if t >= cfg.t_max:
            return Witness("freeze_time", on_labels, acts, X={"timeout": last_reentry})

        r = cfg.residual(cur)
        if verbose:
            print(f"  step t={t}: regimes={on_labels} r(c)={r:.3f} acts={acts}")

        if r <= cfg.theta_r:
            conf = cfg.conflicts(acts)
            if not conf:
                emitted = acts[0] if acts else None
                return Witness("emit", on_labels, acts, emitted=emitted)
            else:
                # conflict: encode and re-enter (here: deterministic, bounded)
                last_reentry = {"conflict": conf}
                t += 1
                # encoding the conflict perturbs content minimally and deterministically;
                # in this demo the conflict is irreducible, so it recurs until timeout.
                continue
        else:
            ok, info = try_expand(cfg, cur)
            if not ok:
                e = cfg.coord_residual(cur)
                Xcoords = [k for k in cfg.active_domain(cur) if e[k] > cfg.tau]
                return Witness("freeze_resid", on_labels, acts, X={"resid_coords": Xcoords})
            else:
                last_reentry = {"expand": info["focus"]}
                t += 1
                # successful expansion would reduce residual; in the abstention demo
                # we construct cases that are genuinely unrepresentable, so 'ok' is False.
                continue


# small helper used by HALT-attribution above
def _act_on_from_labels(cfg: Config, labels):
    L = frozenset(labels)
    acts = set()
    for key, actions in cfg.G.items():
        if frozenset(key) <= L:
            acts.update(actions)
    return sorted(acts)

Config.act_on_from_labels = lambda self, labels: _act_on_from_labels(self, labels)


# --------------------------------------------------------------------------
# Counterfactual simulation: A models B over shared labels with A's own bases.
# --------------------------------------------------------------------------

def simulate_other(cfg_A: Config, cfg_B: Config, c, shared_labels):
    """
    A predicts B's licensed actions using A's regimes restricted to shared labels
    (A's bases), and we compare against B's *real* licensed actions (B's bases).
    Returns (predicted_acts, real_acts, agree).
    """
    # A's prediction: which shared-label regimes A would activate, then license via A's G
    onA = [r.label for r in cfg_A.regime_on(c) if r.label in shared_labels]
    pred = sorted(set(a for a in cfg_A.act_on_from_labels(onA)))
    # B's reality: B's own activation + licensing
    onB = [r.label for r in cfg_B.regime_on(c) if r.label in shared_labels]
    real = sorted(set(a for a in cfg_B.act_on_from_labels(onB)))
    return pred, real, (pred == real)
