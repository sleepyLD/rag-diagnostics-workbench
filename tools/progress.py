"""Progress reporting for runner jobs (read by auto/runner.py's stall watchdog and shown in auto/heartbeat.json).

    from tools.progress import report, start_parent_watch
    report(done, total, stage="eval neoqa/casc")   # cheap; call after every checkpoint
start_parent_watch(): exit if the parent process dies (no orphans left on the GPU)."""
import json, os, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_JOB = os.environ.get("RUNNER_JOB_ID")
_T0 = time.time()


def report(done, total, stage="", **extra):
    if not _JOB:
        return
    d = ROOT / "auto" / "progress"
    d.mkdir(parents=True, exist_ok=True)
    el = time.time() - _T0
    eta = (el / done * (total - done)) if done and total and done < total else None
    rec = {"job": _JOB, "stage": stage, "done": done, "total": total, "pct": round(100 * done / total, 1) if total else None,
           "elapsed_min": round(el / 60, 1), "eta_min": round(eta / 60, 1) if eta is not None else None,
           "updated": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    # Never let progress reporting break the job: on Windows os.replace fails while the runner is reading the file.
    try:
        tmp = d / f"{_JOB}.{os.getpid()}.json.tmp"
        tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        for _ in range(10):
            try:
                os.replace(tmp, d / f"{_JOB}.json")
                return
            except PermissionError:
                time.sleep(0.05)
        tmp.unlink(missing_ok=True)
    except Exception:
        pass


def start_parent_watch(interval=10):
    ppid = os.getppid()

    def watch():
        try:
            import psutil
        except Exception:
            return
        while True:
            time.sleep(interval)
            if not psutil.pid_exists(ppid):
                os._exit(17)
    threading.Thread(target=watch, daemon=True).start()


def ten_percent_batch(n_todo, default):
    """Batch size so that a checkpoint is written at least every ~10% of the work."""
    return max(1, min(default, -(-n_todo // 10))) if n_todo else default


try:   # transformers callback for training scripts
    from transformers import TrainerCallback

    class ProgressCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if state.global_step % 10 == 0 or state.global_step == state.max_steps:
                report(state.global_step, state.max_steps, stage="train")
except Exception:      # transformers not installed (e.g. the CPU test venv)
    ProgressCallback = None
