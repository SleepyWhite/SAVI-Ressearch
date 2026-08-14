#!/usr/bin/env python
"""811 report: layer-wise probe curves, averaged over three fold splits.

Companion to make_figures_811_ci.py (same palette / rcParams / figure size).
Difference: instead of a per-layer cluster-bootstrap CI band on a single fold
split, each curve is the mean over three group->fold assignments and the band
is their min-max range. Source:

  $BELIEF_R_SAVI_ROOT/outputs/ci/probe_layerwise_foldseed.json
  models["<hf-name>"]["splits"]["orig"|"seed1"|"seed2"] -> [bal_acc per layer]
  orig = deterministic GroupKFold; seed1/seed2 = remapped group order
  (probe_foldseed_variant.remap_groups, seeds 20260811 / 20260812)

Instrument check below: the "orig" curve must equal the frozen point estimates
in outputs/layerwise_ci.json, and the run must have recorded its own anchor as
reproduced. Pure CPU. Writes only figures/fig_probe_layerwise_3seed.png; the
existing fig_probe_layerwise.png is NOT touched.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- palette (identical to make_figures_811_ci.py) ----
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK2, "text.color": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# Experiment-output root. Outputs are not shipped with this repo; point
# BELIEF_R_SAVI_ROOT at a local experiment directory (defaults to <repo>).
ROOT = os.environ.get("BELIEF_R_SAVI_ROOT") or os.path.join(HERE, "..", "..")
SEED_JSON = os.path.join(ROOT, "outputs", "ci", "probe_layerwise_foldseed.json")
CI_JSON = os.path.join(ROOT, "outputs", "layerwise_ci.json")

with open(SEED_JSON) as f:
    S = json.load(f)
with open(CI_JSON) as f:
    CI = json.load(f)["models"]

SPLITS = ("orig", "seed1", "seed2")
SERIES = [("Qwen3-4B", "Qwen/Qwen3-4B", BLUE),
          ("Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", ORANGE),
          ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct", AQUA)]

# ---- instrument checks: orig split must be the frozen layer-wise curve ----
for _, hf, _c in SERIES:
    rec = S["models"][hf]
    assert rec["anchor_ok"], (hf, rec["anchor_max_abs_diff"])
    ref = [e["bal_acc"] for e in CI[hf]["layers"]]
    for s in SPLITS:
        assert len(rec["splits"][s]) == len(ref), (hf, s)
    assert max(abs(a - b) for a, b in zip(rec["splits"]["orig"], ref)) <= 5e-4, hf

fig, ax = plt.subplots(figsize=(8.2, 4.6))
for name, hf, c in SERIES:
    cur = [S["models"][hf]["splits"][s] for s in SPLITS]
    n = len(cur[0])
    xs = [i / (n - 1) for i in range(n)]
    mean = [sum(v[i] for v in cur) / len(cur) for i in range(n)]
    los = [min(v[i] for v in cur) for i in range(n)]
    his = [max(v[i] for v in cur) for i in range(n)]
    ax.fill_between(xs, los, his, color=c, alpha=0.13, lw=0)
    ax.plot(xs, mean, color=c, lw=2, label=name, solid_capstyle="round")

# behavioral band: best behavioral acquisition readouts across routes, 0.52-0.61
ax.axhspan(0.519, 0.611, color="#f0efec", zorder=0)
ax.text(0.995, 0.565, "behavioral readouts of the same\ndistinction: 0.52–0.61 (all routes)",
        ha="right", va="center", fontsize=8, color=INK2)
ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)))
ax.text(0.995, 0.503, "chance 0.500", ha="right", fontsize=8, color=INK2)

ax.set_xlim(0, 1)
ax.set_ylim(0.48, 0.80)
ax.set_xlabel("relative depth (layer / n_layers)")
ax.set_ylabel("out-of-fold balanced accuracy")
ax.set_title("Layer-wise probe: mean of three fold splits (band = min–max)")
ax.legend(loc="lower center", frameon=False, fontsize=8, ncol=3,
          handlelength=1.6, columnspacing=1.2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(False, axis="x")
ax.tick_params(length=0)
fig.tight_layout()
path = os.path.join(OUT, "fig_probe_layerwise_3seed.png")
fig.savefig(path, dpi=200)
plt.close(fig)

print("wrote", path)
for name, hf, _c in SERIES:
    cur = [S["models"][hf]["splits"][s] for s in SPLITS]
    n = len(cur[0])
    mean = [sum(v[i] for v in cur) / len(cur) for i in range(n)]
    pk = max(range(n), key=lambda i: mean[i])
    spread = max(max(v[i] for v in cur) - min(v[i] for v in cur) for i in range(n))
    print(f"  {name}: mean-curve peak L{pk} = {mean[pk]:.4f}   max split spread = {spread:.4f}")
