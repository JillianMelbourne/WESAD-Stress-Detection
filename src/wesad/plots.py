"""Plot helpers shared by the notebook's figures."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import COLORS, CONDITION_ORDER, FS_CHEST, LABEL_NAMES, PipelineConfig


def save_fig(fig, name, cfg: PipelineConfig):
    """Save a figure as <fig_dir>/<name>.png when cfg.save_figures is on."""
    if cfg.save_figures:
        cfg.fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")


def condition_periods(label):
    """One row per contiguous block of the same condition (start/end in minutes)."""
    change = np.flatnonzero(np.diff(label)) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(label)]
    return pd.DataFrame({"label": label[starts].astype(int),
                         "start_min": starts / FS_CHEST / 60, "end_min": ends / FS_CHEST / 60})


def shade(ax, periods):
    """Shade an axis (x in minutes) by experimental condition."""
    for _, p in periods[periods["label"].isin(COLORS)].iterrows():
        ax.axvspan(p.start_min, p.end_min, color=COLORS[p.label], alpha=0.35, lw=0)


def condition_legend(ax, labels=tuple(COLORS)):
    """Legend of condition colours placed just above an axis."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k], alpha=0.5) for k in labels]
    ax.legend(handles, [LABEL_NAMES[k] for k in labels], ncol=len(handles), frameon=False,
              loc="lower left", bbox_to_anchor=(0, 1.02))


def condition_boxplots(df, feature_list, title, ncols=4):
    """One boxplot panel per feature, boxes ordered baseline / amusement / stress."""
    nrows = int(np.ceil(len(feature_list) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows), squeeze=False)
    for ax, feat in zip(axes.ravel(), feature_list):
        vals = [df.loc[df["condition"] == c, feat].dropna() for c in CONDITION_ORDER]
        bp = ax.boxplot(vals, patch_artist=True, widths=0.6, medianprops={"color": "black"})
        for patch, lab in zip(bp["boxes"], (1, 3, 2)):
            patch.set_facecolor(COLORS[lab])
        ax.set_xticks([1, 2, 3], CONDITION_ORDER, fontsize=8)
        ax.set_title(feat, fontsize=9)
    for ax in axes.ravel()[len(feature_list):]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def slope_panel(ax, df, col, ylabel):
    """One line per participant from mean non-stress to mean stress value."""
    means = df.groupby(["subject", "stress"])[col].mean().unstack()
    for _, (non, st) in means[[0, 1]].iterrows():
        ax.plot([0, 1], [non, st], color="#999", lw=1, marker="o", ms=4, mfc=COLORS[1])
        ax.plot([1], [st], marker="o", ms=4, color=COLORS[2])
    ax.set_xticks([0, 1], ["non-stress", "stress"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel(ylabel)


def effect_size_chart(d_by_subject, effects):
    """Ranked bars of median within-participant d, participant dots, and pooled normalized d."""
    order = effects.index[::-1]                                   # largest effect at the top
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 1.5))
    y = np.arange(len(order))
    median_d = effects.loc[order, "median within-participant d"]
    ax.barh(y, median_d, color=np.where(median_d > 0, COLORS[2], COLORS[1]), height=0.6)
    rng = np.random.default_rng(0)
    for i, f in enumerate(order):
        d = d_by_subject[f].dropna()
        ax.scatter(d, i + rng.uniform(-0.18, 0.18, len(d)), s=10, color="#555", alpha=0.6, zorder=3)
    ax.scatter(effects.loc[order, "pooled normalized d"], y, marker="D", s=30, color="black", zorder=4,
               label="pooled normalized d")
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xscale("symlog", linthresh=1)                          # linear within ±1, log beyond
    ticks = [-10, -5, -2, -1, 0, 1, 2, 5, 10, 20]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(-12, 25)
    ax.set_yticks(y, order, fontsize=9)
    ax.set_xlabel("Cohen's d, stress vs. non-stress (symmetric log scale)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Which features separate stress from non-stress?")
    fig.tight_layout()
    return fig


def effect_heatmap(d_by_subject, starred=()):
    """Participants x features heatmap of Cohen's d (clipped to ±3); `starred` participants get a *."""
    mat = d_by_subject.clip(-3, 3)
    fig, ax = plt.subplots(figsize=(13, 0.35 * len(mat) + 1.5))
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#e5e5e5")   # grey = not available
    im = ax.imshow(np.ma.masked_invalid(mat.to_numpy()), cmap=cmap, vmin=-3, vmax=3, aspect="auto")
    for (i, j), v in np.ndenumerate(d_by_subject.to_numpy()):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 1.8 else "#222")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]), [f"{s} *" if s in starred else s for s in mat.index], fontsize=8)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cohen's d (clipped to ±3)")
    fig.tight_layout()
    return fig, ax


# ---- single-participant figures ----------------------------------------------------
def aligned_session(sig, periods, sid):
    """Figure 1: chest and wrist signals across the protocol, shaded by condition."""
    panels = [("Chest EDA (µS)", *sig["chest_eda"]),
              ("Wrist EDA (µS)", *sig["wrist_eda"]),
              ("Respiration (filtered)", *sig["resp"]),
              ("Wrist temp (°C)", *sig["wrist_temp"]),
              ("Chest ACC (g)", sig["acc_chest"][0][::175], 4),      # 700 Hz -> 4 Hz for plotting
              ("Wrist ACC (g)", sig["acc_wrist"][0][::8], 4)]        # 32 Hz -> 4 Hz
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 10), sharex=True)
    for ax, (name, y, fs) in zip(axes, panels):
        ax.plot(np.arange(len(y)) / fs / 60, y, lw=0.6, color="#333")
        shade(ax, periods)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
    condition_legend(axes[0])
    axes[-1].set_xlabel("Minutes since start of synced recording")
    fig.suptitle(f"{sid}: chest and wrist signals across the protocol", y=0.995)
    fig.tight_layout()
    return fig


def ecg_vs_bvp(subj, seconds=10, offset_s=60):
    """Figure 2: the same seconds of chest ECG and wrist BVP, one column each for baseline and stress."""
    label = subj["label"]
    signals = [("Chest ECG (mV)", np.ravel(subj["chest"]["ECG"]), FS_CHEST),
               ("Wrist BVP (a.u.)", np.ravel(subj["wrist"]["BVP"]), 64)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 5), sharex="col")
    for col, lab in enumerate([1, 2]):                              # 1 = baseline, 2 = stress
        t0 = np.argmax(label == lab) / FS_CHEST + offset_s          # first sample of the condition + offset
        for row, (name, x, fs) in enumerate(signals):
            start, stop = int(t0 * fs), int((t0 + seconds) * fs)
            axes[row, col].plot(np.arange(start, stop) / fs, x[start:stop], lw=0.8,
                                color=COLORS[lab] if row else "#333")
            axes[row, col].set_ylabel(name if col == 0 else "")
        axes[0, col].set_title(f"{LABEL_NAMES[lab]} ({t0 / 60:.1f} min)")
        axes[1, col].set_xlabel("Seconds since start of synced recording")
    fig.tight_layout()
    return fig


def features_over_time(win, show):
    """Figure 3: window features over time; hollow dots carry at least one quality flag."""
    fig, axes = plt.subplots(len(show), 1, figsize=(13, 1.8 * len(show)), sharex=True, squeeze=False)
    for ax, (col, name) in zip(axes[:, 0], show):
        for lab in (1, 3, 2):                                       # legend order: baseline, amusement, stress
            w = win[win["condition"] == LABEL_NAMES[lab]]
            flagged = w["flag_any"] if "flag_any" in w else pd.Series(False, index=w.index)
            ax.scatter(w.loc[~flagged, "t_start_min"] + 0.5, w.loc[~flagged, col], s=14,
                       color=COLORS[lab], label=LABEL_NAMES[lab])
            ax.scatter(w.loc[flagged, "t_start_min"] + 0.5, w.loc[flagged, col], s=14,
                       facecolor="none", edgecolor=COLORS[lab])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
    axes[0, 0].legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02))
    axes[-1, 0].set_xlabel("Window midpoint (min)")
    fig.tight_layout()
    return fig


# ---- cohort distribution figures ----------------------------------------------------
def condition_histograms(df, feature_list, title, ncols=5, bins=30):
    """Overlaid histograms of each feature, one colour per condition (density-scaled so group sizes compare)."""
    nrows = int(np.ceil(len(feature_list) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False)
    codes = {name: code for code, name in LABEL_NAMES.items() if name in CONDITION_ORDER}
    for ax, feat in zip(axes.ravel(), feature_list):
        values = df[feat].dropna()
        edges = np.histogram_bin_edges(values, bins=bins)               # shared bins so shapes are comparable
        for cond in CONDITION_ORDER:
            v = df.loc[df["condition"] == cond, feat].dropna()
            ax.hist(v, bins=edges, density=True, alpha=0.45, color=COLORS[codes[cond]], label=cond)
        ax.set_title(feat, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
    for ax in axes.ravel()[len(feature_list):]:
        ax.set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def scatter_matrix(df, feature_list, title, max_points=None, seed=0):
    """Grid of pairwise scatterplots (lower triangle) with per-condition histograms on the diagonal."""
    codes = {name: code for code, name in LABEL_NAMES.items() if name in CONDITION_ORDER}
    data = df if max_points is None else df.sample(min(max_points, len(df)), random_state=seed)
    k = len(feature_list)
    fig, axes = plt.subplots(k, k, figsize=(1.9 * k, 1.9 * k))
    for i, fy in enumerate(feature_list):
        for j, fx in enumerate(feature_list):
            ax = axes[i, j]
            if j > i:                                                    # upper triangle left empty
                ax.set_visible(False)
                continue
            for cond in CONDITION_ORDER:
                d = data[data["condition"] == cond]
                if i == j:
                    edges = np.histogram_bin_edges(data[fx].dropna(), bins=25)
                    ax.hist(d[fx].dropna(), bins=edges, density=True, alpha=0.45, color=COLORS[codes[cond]])
                else:
                    ax.scatter(d[fx], d[fy], s=4, alpha=0.35, color=COLORS[codes[cond]], label=cond,
                               rasterized=True)
            ax.tick_params(labelsize=6)
            if i == j:
                ax.set_yticks([])
            if i == k - 1:
                ax.set_xlabel(fx, fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(fy, fontsize=7)
            elif j > 0:
                ax.set_yticklabels([])
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, markerscale=3, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
