# -*- coding: utf-8 -*-
"""
run_all.py - runs stages 01-24 in numerical order, each in a separate Python process.

Usage: open a terminal in this folder (on Windows: Command Prompt or PowerShell) and type one of
  python run_all.py          all stages, 01 to 24
  python run_all.py 13       from stage 13 to the end
  python run_all.py 08 10    stages 08 to 10 only
If "python" does not start Python 3 on your system, use "py" (Windows) or "python3" (macOS/Linux).

Each stage reads the intermediate results that earlier stages saved in OUT_DIR/work, so a partial run
needs those stages to have been run once (stages 13-22 use the results of stages 01, 04, 08 and 10).
If a stage fails, the run stops and prints the command that resumes from that stage.
The input file, the output folder (OUT_DIR) and the NUTS3 option are set at the top of lib.py.
"""
import os
import sys
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
steps = sorted(f for f in os.listdir(HERE) if f[:2].isdigit() and f[2] == "_" and f.endswith(".py"))
first = sys.argv[1].zfill(2) if len(sys.argv) > 1 else steps[0][:2]
last = sys.argv[2].zfill(2) if len(sys.argv) > 2 else steps[-1][:2]
todo = [s for s in steps if first <= s[:2] <= last]
if not todo:
    sys.exit(f"No steps between {first} and {last}. Available: {', '.join(s[:2] for s in steps)}")

env = dict(os.environ, PYTHONIOENCODING="utf-8")
t_all = time.time()
for s in todo:
    t0 = time.time()
    print(f"\n>>> {s}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=HERE, env=env)
    if r.returncode != 0:
        print(f"\n!!! {s} stopped with an error (exit code {r.returncode}). After fixing it, resume with: python run_all.py {s[:2]}")
        sys.exit(r.returncode)
    print(f"<<< {s} finished in {time.time() - t0:.0f} s", flush=True)
print(f"\nDone: {len(todo)} steps in {(time.time() - t_all) / 60:.1f} min. Results are in the OUT_DIR folder set at the top of lib.py.")
