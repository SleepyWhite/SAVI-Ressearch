#!/usr/bin/env python
"""Figures for REPORT.md (2026-08-05).

All numbers are hardcoded FROM the frozen recomputation files under outputs/ci/
(provenance noted per block). Rerun any ci script to re-derive them; this script
only draws. Pure CPU. Labels are English because the host has no CJK font;
captions in REPORT.md carry the Chinese reading.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- palette (dataviz reference instance, validated 3-slot all-pairs) ----
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)


def style(ax, xgrid=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x" if xgrid else "y")
    ax.grid(False, axis="y" if xgrid else "x")
    ax.tick_params(length=0)


# =====================================================================
# Fig 1 — layer-wise probe curves
# provenance: outputs/ci/hidden_probe.txt (description block: per-layer fixed-layer OOF bal-acc)
#             outputs/ci/hidden_probe_llama.txt (same block)
# =====================================================================
q3_4b = [0.500, 0.590, 0.612, 0.677, 0.678, 0.682, 0.665, 0.684, 0.718, 0.697,
         0.696, 0.695, 0.708, 0.710, 0.708, 0.727, 0.706, 0.706, 0.729, 0.735,
         0.725, 0.701, 0.702, 0.690, 0.688, 0.658, 0.679, 0.688, 0.669, 0.660,
         0.679, 0.675, 0.677, 0.688, 0.698, 0.696, 0.688]
q25_7b = [0.500, 0.564, 0.580, 0.598, 0.621, 0.653, 0.654, 0.682, 0.701, 0.707,
          0.729, 0.699, 0.726, 0.726, 0.728, 0.705, 0.694, 0.724, 0.712, 0.726,
          0.722, 0.687, 0.690, 0.682, 0.666, 0.654, 0.671, 0.673, 0.652]
llama = [0.500, 0.601, 0.623, 0.666, 0.701, 0.680, 0.718, 0.713, 0.722, 0.739,
         0.739, 0.709, 0.718, 0.714, 0.685, 0.714, 0.712, 0.729, 0.722, 0.682,
         0.676, 0.716, 0.678, 0.678, 0.659, 0.680, 0.678, 0.659, 0.660, 0.675,
         0.662, 0.664, 0.669]

fig, ax = plt.subplots(figsize=(8.2, 4.6))
series = [("Qwen3-4B  (peak L19 = 0.735)", q3_4b, BLUE, 19),
          ("Qwen2.5-7B  (peak L10 = 0.729)", q25_7b, ORANGE, 10),
          ("Llama-3.1-8B  (peak L9 = 0.739)", llama, AQUA, 9)]
for name, ys, c, peak in series:
    xs = [i / (len(ys) - 1) for i in range(len(ys))]
    ax.plot(xs, ys, color=c, lw=2, label=name, solid_capstyle="round")
    ax.plot(xs[peak], ys[peak], "o", color=c, ms=6, mec=SURFACE, mew=1.5)
# behavioral band: best behavioral acquisition readouts across routes, 0.52-0.61
ax.axhspan(0.519, 0.611, color="#f0efec", zorder=0)
ax.text(0.995, 0.565, "behavioral readouts of the same\ndistinction: 0.52–0.61 (all routes)",
        ha="right", va="center", fontsize=8, color=INK2)
ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)))
ax.text(0.995, 0.503, "chance 0.500", ha="right", fontsize=8, color=INK2)
ax.text(0.012, 0.507, "embedding layer = 0.500 by construction", fontsize=8,
        color=INK2, va="bottom")
ax.set_xlim(0, 1); ax.set_ylim(0.48, 0.78)
ax.set_xlabel("relative depth (layer / n_layers)")
ax.set_ylabel("out-of-fold balanced accuracy")
ax.set_title("REQ/ALT is linearly readable in mid layers, and fades toward the output\n"
             r"(L7 linear probe, lv=5 ponens, grouped 5-fold CV, N=1,744 rows)")
ax.legend(loc="lower center", frameon=False, fontsize=8, ncol=3,
          handlelength=1.6, columnspacing=1.2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_probe_layerwise.png"), dpi=200)
plt.close(fig)

# =====================================================================
# Fig 2 — one-ruler pathway ladder (all on the 4B pipeline, lv=5 ponens bal-acc)
# provenance: outputs/ci/b3.txt (C anchor 0.5189, B3 0.6407 [.594,.689])
#             outputs/ci/b4.txt via STATUS 6.21 (B4 0.6306 [.581,.680])
#             outputs/ci/b4long.txt (B4L 0.6231 [.574,.674]; last-layer probe 0.688)
#             outputs/ci/b3long.txt (B3L 0.7132 [.666,.760], folds 0.659–0.782)
#             outputs/ci/hidden_probe.txt (probe 0.692 [.636,.744])
#             outputs/layerwise_ci.json (L36 CI [.6294,.7432], PREREG_layerwise_ci)
#             outputs/ci/ceiling.txt (0.843); STATUS 6.14b (human ~0.890, estimate)
# 2026-08-12: rows re-paired (budget-x4 arm directly under its base arm), B3L added,
#             L36 CI added, plain-English title — mirrors report/811 figure exactly.
# =====================================================================
rows = [  # label, value, lo, hi, kind, note
    ("Human single annotator (estimate)", 0.890, None, None, "ref", ""),
    ("Intent ceiling (lv=5)", 0.843, None, None, "ref", ""),
    ("Linear probe, selected mid layer", 0.692, 0.636, 0.744, "m", "no collapse; folds 0.60–0.76"),
    ("Linear probe @ last layer L36", 0.688, 0.6294, 0.7432, "m", ""),
    ("B3  deep LoRA (all-linear)", 0.641, 0.594, 0.689, "m", "folds 0.50–0.78, unstable"),
    ("B3L  deep LoRA, budget ×4", 0.713, 0.666, 0.760, "m", "converged; folds 0.66–0.78"),
    ("B4  last-layer LoRA (lm_head)", 0.631, 0.581, 0.680, "m", "clean dynamics"),
    ("B4L  last-layer LoRA, budget ×4", 0.623, 0.574, 0.674, "m", "converged; residual real"),
    ("C  prompt self-judgement", 0.519, None, None, "m", ""),
    ("Chance", 0.500, None, None, "ref", ""),
]
fig, ax = plt.subplots(figsize=(8.2, 4.8))
ys = range(len(rows) - 1, -1, -1)
for y, (label, v, lo, hi, kind, note) in zip(ys, rows):
    c = BLUE if kind == "m" else INK2
    if lo is not None:
        ax.plot([lo, hi], [y, y], color=c, lw=2, alpha=0.45, solid_capstyle="round")
    ax.plot(v, y, "o", color=c, ms=8 if kind == "m" else 7,
            mfc=c if kind == "m" else SURFACE, mec=c, mew=1.5)
    ax.text(v, y + 0.32, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    if note:
        ax.text(0.998, y, note, fontsize=7.5, color=INK2, va="center", ha="right")
ax.set_yticks(list(ys))
ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_xlim(0.45, 1.0); ax.set_ylim(-0.6, len(rows) - 0.2)
ax.set_xlabel("balanced accuracy on the REQ/ALT distinction (lv=5 ponens, one ruler)")
ax.set_title("Conditional-relation judgment accuracy across methods (Qwen3-4B)")
style(ax, xgrid=True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_pathway_ladder.png"), dpi=200)
plt.close(fig)

# =====================================================================
# Fig 3 — end-to-end repair by state source (BREU, lv=5, n=391 scenes)
# provenance: outputs/ci/probe_decode.txt (A/B1/B2/C + greedy)
#             outputs/ci/b3.txt (B3 0.6046), outputs/ci/b4long.txt (B4L 0.5465)
# =====================================================================
arms = [  # label, BREU, lo, hi, family
    ("A  probe + inject (4B)", 0.6557, 0.6140, 0.6953, "rep"),
    ("A  probe + inject (7B)", 0.6181, 0.5825, 0.6549, "rep"),
    ("B3  structured LoRA (deep)", 0.6046, 0.5613, 0.6476, "grad"),
    ("B4L  last-layer LoRA head", 0.5465, 0.5084, 0.5846, "grad"),
    ("B1  LoRA all-linear (end-to-end)", 0.5188, 0.4961, 0.5424, "grad"),
    ("B2  LoRA layers 20–36", 0.5161, 0.5002, 0.5329, "grad"),
    ("C  two-step self-classify", 0.4869, 0.4454, 0.5269, "prompt"),
]
fam_color = {"rep": BLUE, "grad": ORANGE, "prompt": AQUA}
fig, ax = plt.subplots(figsize=(8.2, 4.2))
ys = range(len(arms) - 1, -1, -1)
for y, (label, v, lo, hi, fam) in zip(ys, arms):
    c = fam_color[fam]
    ax.plot([lo, hi], [y, y], color=c, lw=2, alpha=0.45, solid_capstyle="round")
    ax.plot(v, y, "o", color=c, ms=8, mec=SURFACE, mew=1)
    ax.text(hi + 0.006, y, f"{v:.3f}", va="center", fontsize=8, color=INK)
ax.axvline(0.4928, color=INK2, lw=1, ls=(0, (4, 3)))
ax.text(0.4928, len(arms) - 0.25, "greedy 0.493", ha="center", fontsize=8, color=INK2)
ax.set_yticks(list(ys)); ax.set_yticklabels([a[0] for a in arms], fontsize=8.5)
ax.set_xlim(0.42, 0.74); ax.set_ylim(-0.6, len(arms) - 0.2)
ax.set_xlabel("BREU (lv=5, n=391 scenes; whiskers = scene-clustered bootstrap 95% CI)")
ax.set_title("Same injection machinery, three state sources:\nonly the representation read-out repairs fully")
handles = [plt.Line2D([], [], marker="o", ls="", color=fam_color[f], ms=7)
           for f in ("rep", "grad", "prompt")]
ax.legend(handles, ["state from representation (probe)", "state from gradient (LoRA)",
                    "state from prompting"], loc="lower right", frameon=False, fontsize=8)
style(ax, xgrid=True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_repair_breu.png"), dpi=200)
plt.close(fig)

# =====================================================================
# Fig 4 — dose-response across agreement levels
# provenance: outputs/ci/lv4_dose.txt (dose-response table)
# =====================================================================
dose = [  # label, d5, lo5, hi5, d4, lo4, hi4, color
    ("A  probe + inject", 0.1629, 0.1226, 0.2017, 0.0848, 0.0490, 0.1207, BLUE),
    ("B4L  last-layer head", 0.0537, 0.0142, 0.0930, 0.0006, -0.0348, 0.0364, ORANGE),
    ("C  prompt self-judge", -0.0058, -0.0463, 0.0344, 0.0467, 0.0126, 0.0803, AQUA),
]
fig, ax = plt.subplots(figsize=(6.8, 4.6))
xoff = {"A  probe + inject": -0.03, "B4L  last-layer head": 0.0, "C  prompt self-judge": 0.03}
for label, d5, lo5, hi5, d4, lo4, hi4, c in dose:
    x = [0 + xoff[label], 1 + xoff[label]]
    ax.plot(x, [d5, d4], color=c, lw=2, marker="o", ms=7, mec=SURFACE, mew=1)
    for xi, (lo, hi) in zip(x, [(lo5, hi5), (lo4, hi4)]):
        ax.plot([xi, xi], [lo, hi], color=c, lw=2, alpha=0.45, solid_capstyle="round")
    ax.annotate(f"{d5:+.3f}", (x[0], d5), textcoords="offset points",
                xytext=(-8, 4), ha="right", fontsize=8, color=INK)
    ax.annotate(f"{d4:+.3f}", (x[1], d4), textcoords="offset points",
                xytext=(8, 4), ha="left", fontsize=8, color=INK)
    ylab = {"A  probe + inject": d4, "C  prompt self-judge": d4 + 0.012,
            "B4L  last-layer head": d4 - 0.016}[label]
    ax.text(1.24, ylab, label, va="center", fontsize=8.5, color=c)
ax.axhline(0, color=INK2, lw=1, ls=(0, (4, 3)))
ax.set_xticks([0, 1])
ax.set_xticklabels(["lv=5  (probe signal 0.692,\nintent ceiling 0.843)",
                    "lv=4  (probe signal 0.604,\nintent ceiling 0.728)"])
ax.set_xlim(-0.35, 2.05); ax.set_ylim(-0.08, 0.23)
ax.set_ylabel("ΔBREU vs greedy (paired, 95% CI)")
ax.set_title("Supervised arms shrink with the signal;\nthe zero-shot arm does not (pre-registered surprise)")
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_dose_response.png"), dpi=200)
plt.close(fig)

# =====================================================================
# Fig 5 — per-fold spread, LoRA pathways vs probe
# provenance: outputs/ci/b4.txt line 19 (B3 same-estimator folds, range 0.2778)
#             outputs/ci/b4long.txt (B4/B4L folds); pooled values from b3/b4/b4long
#             outputs/ci/b3long.txt line 45 (B3L folds, range 0.1238, pooled 0.7132)
#             probe folds computed 2026-08-05 from outputs/probe_decode/
#             A_Qwen_Qwen3_4B.jsonl (`fold` field; gold from ground_truth,
#             ponens REQ<=>'c'; pooled anchor 0.6917 reproduced exactly)
# =====================================================================
folds = {
    "B3  deep LoRA": ([0.5718, 0.5000, 0.7778, 0.6386, 0.7244], 0.6407, ORANGE),
    "B3L  deep ×4": ([0.7217, 0.6901, 0.7824, 0.6586, 0.7048], 0.7132, ORANGE),
    "B4  last-layer": ([0.590, 0.704, 0.690, 0.596, 0.611], 0.6306, ORANGE),
    "B4L  budget ×4": ([0.580, 0.749, 0.644, 0.586, 0.594], 0.6231, ORANGE),
    "Probe (arm A)": ([0.6955, 0.6047, 0.7245, 0.6643, 0.7614], 0.6917, BLUE),
}
fig, ax = plt.subplots(figsize=(7.6, 4.4))
for i, (name, (vals, pooled, c)) in enumerate(folds.items()):
    xs = [i + (j - 2) * 0.07 for j in range(5)]
    ax.plot(xs, vals, "o", color=c, ms=7, mec=SURFACE, mew=1, alpha=0.85)
    ax.plot([i - 0.2, i + 0.2], [pooled, pooled], color=INK, lw=2,
            solid_capstyle="round")
    ax.text(i, max(vals) + 0.014, f"range {max(vals)-min(vals):.3f}",
            ha="center", fontsize=8, color=INK2)
    ax.text(i, min(vals) - 0.026, f"pooled {pooled:.3f}", ha="center",
            fontsize=8, color=INK)
ax.axhline(0.519, color=INK2, lw=1, ls=(0, (4, 3)))
handles = [
    plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=7,
               label="one fold (gradient / LoRA)"),
    plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=7,
               label="one fold (probe read-out)"),
    plt.Line2D([], [], color=INK, lw=2, label="pooled over 5 folds"),
    plt.Line2D([], [], color=INK2, lw=1, ls=(0, (4, 3)),
               label="prompt self-judge 0.519"),
]
ax.legend(handles=handles, loc="lower right", fontsize=7.5, frameon=True,
          facecolor=SURFACE, edgecolor=GRID, framealpha=1.0)
ax.set_xticks(range(len(folds))); ax.set_xticklabels(list(folds.keys()), fontsize=9)
ax.set_xlim(-0.5, len(folds) - 0.35); ax.set_ylim(0.43, 0.82)
ax.set_ylabel("per-fold out-of-fold balanced accuracy")
ax.set_title("Small-budget deep LoRA collapses (one fold stuck at 0.50);\n"
             "with budget ×4 every fold converges and climbs out (B3L)")
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_fold_stability.png"), dpi=200)
plt.close(fig)

# =====================================================================
# Fig 6 — probe sample efficiency
# provenance: outputs/ci/probe_decode_sample_curve.txt (5 resamples, mean [min,max])
# =====================================================================
scenes = [50, 100, 200, 313]
eff = {
    "Qwen3-4B": ([0.6111, 0.6492, 0.6867, 0.6917],
                 [0.5685, 0.6138, 0.6694, 0.6917],
                 [0.6505, 0.6805, 0.7051, 0.6917], BLUE),
    "Qwen2.5-7B": ([0.6033, 0.6438, 0.6821, 0.7158],
                   [0.5588, 0.6219, 0.6492, 0.7158],
                   [0.6547, 0.6732, 0.6969, 0.7158], ORANGE),
}
fig, ax = plt.subplots(figsize=(6.8, 4.2))
for name, (mean, lo, hi, c) in eff.items():
    ax.plot(scenes, mean, color=c, lw=2, marker="o", ms=7, mec=SURFACE, mew=1,
            label=name)
    ax.fill_between(scenes, lo, hi, color=c, alpha=0.12, lw=0)
    ax.text(scenes[-1] + 6, mean[-1], f"{mean[-1]:.3f}", va="center", fontsize=8,
            color=INK)
ax.axhspan(0.519, 0.611, color="#f0efec", zorder=0)
ax.text(316, 0.565, "behavioral\nband", ha="left", va="center", fontsize=8, color=INK2)
ax.set_xticks(scenes)
ax.set_xticklabels(["50", "100", "200", "313\n(full set,\nsingle fit)"])
ax.set_xlim(35, 355); ax.set_ylim(0.5, 0.75)
ax.set_xlabel("gold-labelled training scenes per fold (group subsampled)")
ax.set_ylabel("out-of-fold balanced accuracy")
ax.set_title("Probe sample efficiency: 50 scenes already clear the behavioral band\n(band = 5 resamples min–max)")
ax.legend(loc="lower right", frameon=False, fontsize=8.5)
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_sample_efficiency.png"), dpi=200)
plt.close(fig)

print("wrote 6 figures to", OUT)
