"""Standard results-JSON writer with provenance.

Golden rule of this suite: a number only counts if it is in a results JSON that
a GPU/box actually produced. This module is the single writer; it stamps every
result with enough provenance to trace it back to the exact code and machine.

Schema (one file per method x dataset run):
{
  "method": "DeSE", "family": "community_detection", "dataset": "Cora",
  "upstream_repo": "...", "upstream_commit": "6292c0c",
  "harness_git": "<sha of survey repo>", "instance": "<vast id>",
  "host": "<hostname>", "gpu": "<name or null>",
  "started": "<iso>", "finished": "<iso>", "wallclock_sec": 123.4,
  "seeds": [0,1,2,3,4],
  "metrics": {                # raw per-seed lists, NEVER just means
     "ari": [..per seed..], "nmi": [...], ...
  },
  "summary": { "ari": {"mean":.., "ci95":..}, ... },   # convenience, derived
  "notes": "...", "deviations": "any change from upstream defaults"
}
"""
from __future__ import annotations
import json, os, math, socket, subprocess, datetime
from typing import Any


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _git_sha(path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def mean_ci95(xs: list[float]) -> dict[str, float]:
    xs = [float(x) for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "n": 0}
    m = sum(xs) / n
    if n == 1:
        return {"mean": m, "ci95": 0.0, "n": 1}
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    # 1.96 normal approx; small-n note left to the reader
    return {"mean": m, "ci95": 1.96 * sd / math.sqrt(n), "n": n}


class RunResult:
    def __init__(self, method: str, family: str, dataset: str,
                 upstream_repo: str | None = None, upstream_commit: str | None = None,
                 harness_path: str = ".", instance: str | None = None,
                 gpu: str | None = None, notes: str = "", deviations: str = ""):
        self.d: dict[str, Any] = {
            "method": method, "family": family, "dataset": dataset,
            "upstream_repo": upstream_repo, "upstream_commit": upstream_commit,
            "harness_git": _git_sha(harness_path),
            "instance": instance or os.environ.get("VAST_INSTANCE_ID"),
            "host": socket.gethostname(), "gpu": gpu,
            "started": _now(), "finished": None, "wallclock_sec": None,
            "seeds": [], "metrics": {}, "summary": {},
            "notes": notes, "deviations": deviations,
        }
        self._t0 = datetime.datetime.now()

    def add_seed(self, seed: int, **metrics: float) -> None:
        self.d["seeds"].append(seed)
        for k, v in metrics.items():
            self.d["metrics"].setdefault(k, []).append(None if v is None else float(v))

    def finalize(self) -> dict[str, Any]:
        self.d["finished"] = _now()
        self.d["wallclock_sec"] = (datetime.datetime.now() - self._t0).total_seconds()
        self.d["summary"] = {k: mean_ci95(v) for k, v in self.d["metrics"].items()}
        return self.d

    def write(self, results_dir: str) -> str:
        self.finalize()
        os.makedirs(results_dir, exist_ok=True)
        date = datetime.date.today().isoformat()
        fn = f"{self.d['method']}__{self.d['dataset']}__{date}.json"
        path = os.path.join(results_dir, fn)
        with open(path, "w") as f:
            json.dump(self.d, f, indent=2)
        return path
