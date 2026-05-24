"""Run all three demonstrations and regenerate the figure. Deterministic."""
import subprocess, sys
for m in ["demo_abstention", "demo_empathy", "demo_prerequisites"]:
    print("\n" + "#" * 72)
    subprocess.run([sys.executable, f"{m}.py"], check=True)
print("\n" + "#" * 72)
subprocess.run([sys.executable, "make_figure.py"], check=True)
