"""Run every dataset builder; one failure does not stop the others.   python -m tools.bench.prepare_all [names...]"""
import json, subprocess, sys, time
from pathlib import Path
JOBS = {"hotpotqa": ["tools.bench.prepare_hotpot_like", "hotpotqa"], "2wikimultihopqa": ["tools.bench.prepare_hotpot_like", "2wikimultihopqa"],
        "musique": ["tools.bench.prepare_musique"], "multihop_rag": ["tools.bench.prepare_multihop_rag"],
        "neoqa": ["tools.bench.prepare_neoqa", "--key", "23456"], "squad": ["tools.bench.prepare_squad"], "scifact": ["tools.bench.prepare_scifact"]}
status = {}
for name in (sys.argv[1:] or list(JOBS)):
    t0 = time.time()
    rc = subprocess.run([sys.executable, "-m", *JOBS[name]]).returncode
    status[name] = {"returncode": rc, "seconds": round(time.time() - t0, 1)}
    print(f"##### {name}: rc={rc} {status[name]['seconds']}s", flush=True)
Path("bench/data").mkdir(parents=True, exist_ok=True)
Path("bench/data/_prepare_status.json").write_text(json.dumps(status, indent=1))
sys.exit(0 if all(s["returncode"] == 0 for s in status.values()) else 1)
