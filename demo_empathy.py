"""
Demonstration 2 -- Bounded empathy

Two agents share the label "pain" but weight its two axes (injury vs
psychosomatic) differently -- their private bases. Each reads content through
its OWN weighting and licenses 'imaging' (injury-dominant) or 'referral'
(psychosomatic-dominant). A models B with A's weighting (no access to B's).

Faithful phenomenon:
  * clear-cut content (strongly one axis) -> both agents agree
  * ambiguous content (mixed)             -> readings diverge
  * divergence is localized to 'pain' and invisible to A until actions differ.
"""
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# pain content c = (injury_signal, psychosomatic_signal)
# Agent basis = how strongly it weights each axis when reading "pain".
wA = np.array([1.3, 0.7])   # A: injury-leaning frame
wB = np.array([0.7, 1.3])   # B: psychosomatic-leaning frame

def angle_between(u, v):
    c = abs(float(u @ v)) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(min(1.0, max(-1.0, c))))

def action(w, c):
    injury = w[0] * c[0]
    psycho = w[1] * c[1]
    return "imaging" if injury >= psycho else "referral"

print("=" * 70); print("DEMONSTRATION 2 -- BOUNDED EMPATHY"); print("=" * 70)
print(f"\nShared label 'pain'; private weightings differ by angle "
      f"{angle_between(wA, wB):.1f} deg")

rows, agree = [], []
for a in np.linspace(0.0, 1.0, 21):
    c = np.array([1.0 - a, a])
    aA, aB = action(wA, c), action(wB, c)
    rows.append((c.copy(), aA, aB, aA == aB)); agree.append(aA == aB)

print("\nContent sweep (A predicts B with A's weighting; B acts with B's):")
print(f"  {'content':>14}  {'A pred':>9}  {'B real':>9}  match")
for c, aA, aB, ok in rows[::2]:
    print(f"  {str(np.round(c,2)):>14}  {aA:>9}  {aB:>9}  {'OK' if ok else 'DIVERGE'}")

na = sum(agree)
div = [i for i, ok in enumerate(agree) if not ok]
print(f"\nActions agree on {na}/{len(agree)} sampled contents; diverge on {len(agree)-na}.")
if div:
    lo = rows[div[0]][0]; hi = rows[div[-1]][0]
    print(f"Divergence appears only in the ambiguous band {np.round(lo,2)} ... {np.round(hi,2)};")
    print("clear-cut injury or psychosomatic content yields agreement.")
print("On agreeing content A records confirmation -- the divergence leaves no")
print("trace until an ambiguous case makes the actions differ. A's prediction")
print("depends on A's weighting alone, so the error is forced, self-invisible,")
print("and localized to the single shared label 'pain'.")

RESULTS = {"angle_deg": angle_between(wA, wB),
           "sweep": [(float(c[0]), float(c[1]), aA, aB, ok) for c, aA, aB, ok in rows]}
