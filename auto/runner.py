"""Job-queue runner (runs on the Windows host, inside .venv-auto).

Claude drops job files into auto/queue/*.json; this script runs them one at a time
with THIS venv's python, from the project root, and writes logs + results back.
It only ever launches `python <argv...>` — nothing else.

Job file:  {"id": "010_env_check", "argv": ["auto/jobs/env_check.py"], "timeout_s": 3600}
           argv may also be ["-m", "pip", "install", ...]
Outputs:   auto/logs/<id>.log, auto/done/<id>.json, auto/heartbeat.json
Stop:      create an empty file auto/STOP (or just close the window).
Crash safety: a job that was running when the runner died (window closed, power loss, BSOD) is put back at the head of
the queue on the next start, up to MAX_ATTEMPTS times in total; its log is appended, not overwritten.
"""
import json, os, shutil, socket, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUTO = ROOT / "auto"
Q, RUN, DONE, LOGS = AUTO / "queue", AUTO / "running", AUTO / "done", AUTO / "logs"
for d in (Q, RUN, DONE, LOGS):
    d.mkdir(parents=True, exist_ok=True)
MAX_ATTEMPTS = 3


def kill_tree(pid):
    """Kill a process and all its descendants (p.kill() alone leaves grandchildren, e.g. run_matrix -> run_retrieval, on the GPU)."""
    try:
        import psutil
        root = psutil.Process(int(pid))
        procs = root.children(recursive=True) + [root]
        for pr in procs:
            try:
                pr.kill()
            except Exception:
                pass
        psutil.wait_procs(procs, timeout=30)
        return
    except Exception:
        pass
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=60)
        else:
            os.kill(int(pid), 9)
    except Exception:
        pass


def pid_alive(pid):
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        return False


def read_progress(jid):
    try:   # tolerant: the job may be replacing the file right now
        return json.loads((AUTO / "progress" / f"{jid}.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        return out or None
    except Exception as e:
        return f"nvidia-smi failed: {e}"


def heartbeat(job=None, started=None):
    hb = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "epoch": time.time(), "pid": os.getpid(),
          "host": socket.gethostname(), "python": sys.version.split()[0], "gpu": gpu_info(),
          "job": job, "job_started_epoch": started, "progress": read_progress(job) if job else None,
          "queued": sorted(p.name for p in Q.glob("*.json"))}
    tmp = AUTO / "heartbeat.json.tmp"
    tmp.write_text(json.dumps(hb, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, AUTO / "heartbeat.json")


def run_job(path: Path):
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
        jid, argv = str(job["id"]), job["argv"]
        assert isinstance(argv, list) and all(isinstance(a, str) for a in argv) and argv
    except Exception as e:
        (DONE / (path.stem + ".json")).write_text(json.dumps({"id": path.stem, "error": f"bad job file: {e}"}), encoding="utf-8")
        path.unlink(missing_ok=True)
        return
    job["attempt"] = int(job.get("attempt", 0)) + 1
    running = RUN / path.name
    tmp = RUN / (path.name + ".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")   # running/ keeps the attempt count
    os.replace(tmp, running)
    path.unlink(missing_ok=True)
    (AUTO / "progress").mkdir(exist_ok=True)
    stall_s = float(job.get("stall_minutes", 40)) * 60      # no log output AND no progress update for this long -> stuck
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1", RUNNER_ATTEMPT=str(job["attempt"]),
               RUNNER_JOB_ID=jid,
               **{str(k): str(v) for k, v in (job.get("env") or {}).items()})
    timeout = float(job.get("timeout_s", 6 * 3600))
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] start {jid}: python {' '.join(argv)}", flush=True)
    status = "ok"
    with open(LOGS / f"{jid}.log", "a" if job["attempt"] > 1 else "w", encoding="utf-8", errors="replace") as log:
        if job["attempt"] > 1:
            log.write(f"\n===== attempt {job['attempt']} ({time.strftime('%Y-%m-%d %H:%M:%S')}) =====\n"); log.flush()
        p = subprocess.Popen([sys.executable, *argv], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        job["pid"] = p.pid                                     # lets a restarted runner kill an orphan of this job
        tmp = RUN / (path.name + ".tmp"); tmp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8"); os.replace(tmp, running)
        prog_file = AUTO / "progress" / f"{jid}.json"
        while p.poll() is None:
            heartbeat(jid, t0)
            now = time.time()
            last = max([t0] + [f.stat().st_mtime for f in (LOGS / f"{jid}.log", prog_file) if f.exists()])
            if now - t0 > timeout:
                status = "timeout"; kill_tree(p.pid); break
            if (AUTO / "STOP").exists():
                status = "stopped"; kill_tree(p.pid); break
            if now - last > stall_s:
                status = "stalled"
                log.write(f"\n[watchdog] no log output or progress for {stall_s / 60:.0f} min -> killed (stuck / infinite loop?)\n"); log.flush()
                kill_tree(p.pid); break
            time.sleep(5)
        try:
            p.wait(timeout=60)
        except Exception:
            pass
    res = {"id": jid, "argv": argv, "attempt": job["attempt"], "returncode": p.returncode, "status": status if p.returncode == 0 or status != "ok" else "failed",
           "started": t0, "ended": time.time(), "duration_s": round(time.time() - t0, 1)}
    res["progress_at_end"] = read_progress(jid)
    if status == "stalled" and job["attempt"] < MAX_ATTEMPTS:       # retry once more; evaluation / training resume from checkpoints
        job.pop("pid", None)
        tq = Q / (path.name + ".tmp"); tq.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8"); os.replace(tq, Q / path.name)
        res["status"] = "stalled_requeued"
    tmp = DONE / f"{jid}.json.tmp"
    tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, DONE / f"{jid}.json")
    running.unlink(missing_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] done  {jid}: {res['status']} rc={p.returncode} {res['duration_s']}s", flush=True)


def main():
    print(f"runner up. root={ROOT}\nwaiting for jobs in {Q} (create auto/STOP to quit)", flush=True)
    for stale in RUN.glob("*.json"):          # a job interrupted by a previous shutdown / crash
        try:
            job = json.loads(stale.read_text(encoding="utf-8"))
        except Exception:
            job = {"id": stale.stem, "attempt": MAX_ATTEMPTS}
        note = {"id": job.get("id", stale.stem), "attempt": job.get("attempt", 1), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        if job.get("pid") and pid_alive(job["pid"]):
            kill_tree(job["pid"]); note["killed_orphan_pid"] = job["pid"]
        job.pop("pid", None)
        if int(job.get("attempt", 1)) < MAX_ATTEMPTS:
            tmp = Q / (stale.name + ".tmp")
            tmp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, Q / stale.name)    # same file name -> same place in the queue order
            stale.unlink(missing_ok=True)
            note["action"] = "requeued"
        else:
            shutil.move(str(stale), str(DONE / (stale.stem + ".interrupted.json")))
            note["action"] = f"gave up after {MAX_ATTEMPTS} attempts"
        print(f"interrupted job {note}", flush=True)
        with open(AUTO / "recovery_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(note, ensure_ascii=False) + "\n")
    while not (AUTO / "STOP").exists():
        heartbeat()
        jobs = sorted(Q.glob("*.json"))
        if jobs:
            run_job(jobs[0])
        else:
            time.sleep(5)
    print("STOP file found, exiting.")


if __name__ == "__main__":
    main()
