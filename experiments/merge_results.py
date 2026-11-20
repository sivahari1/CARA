"""Merge the separately computed baselines into the main result files (run after synthetic_main.py,
real_notredame.py, external_baselines.py and revision_rerun.py):
  - GLOMAP-RA and Shonan-H (external_*.jsonl) are added to 'methods';
  - CycFilt v2 (revision_*.jsonl) replaces the first CycFilt version, and the matched-camera fields of
    revision_real.jsonl are copied into real_nd.jsonl. The pre-merge files are kept as *.v1.jsonl."""
import os, json, shutil
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
KEYS = ("cara_p95_reg", "cg_p95_reg", "cara_med_reg", "cg_med_reg", "cg_p95_abst", "cg_frac_ok_abst")


def load(n):
    return [json.loads(l) for l in open(os.path.join(RES, n))]


def merge(main, ext, rev):
    rows = load(main); key = lambda r: (r["seed"], r["cond"], r["scene"])
    E = {key(r): r for r in load(ext)}; V = {key(r): r for r in load(rev)}
    v1 = os.path.join(RES, main.replace(".jsonl", ".v1.jsonl"))
    if not os.path.exists(v1):
        shutil.copy(os.path.join(RES, main), v1)
    for r in rows:
        k = key(r)
        for m in ("GLOMAP-RA", "Shonan-H"):
            if k in E: r["methods"][m] = E[k][m]
        if k in V:
            r["methods"]["CycFilt"] = V[k]["CycFilt"]
            for f in KEYS:
                if f in V[k]: r[f] = V[k][f]
    with open(os.path.join(RES, main), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(main, len(rows), "rows merged")


merge("synth_main.jsonl", "external_synth.jsonl", "revision_synth_cycfilt.jsonl")
merge("real_nd.jsonl", "external_real.jsonl", "revision_real.jsonl")
