#!/usr/bin/env python3
"""
KG Snapshot Analysis
Analyzes all 96 KG episode snapshots for thesis evidence.

Produces 4 figures saved to data/kg_analysis/:
  1. relation_type_distribution.png  — handover_ready vs tool_near per episode
  2. object_cooccurrence_heatmap.png — tool_near object-pair frequency matrix
  3. kg_activity_timeline.png        — relation composition over recording dates
  4. episode_complexity.png          — nodes/relations count distribution

Run from repo root:
    python3 scripts/analyze_kg_snapshots.py
"""

import json
import glob
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

# ── paths ────────────────────────────────────────────────────────────────────
SNAPSHOT_DIR = Path("data/kg_snapshots")
OUTPUT_DIR   = Path("data/kg_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSEMBLY_OBJECTS = [
    "platform", "bracket", "cylinder", "pi",
    "screw_big", "screw_small", "nut", "hex_wrench",
]

# ── load all snapshots ────────────────────────────────────────────────────────
def load_snapshots():
    files = sorted(SNAPSHOT_DIR.glob("kg_episode_*.json"))
    episodes = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            d["_file"] = f.name
            episodes.append(d)
        except Exception as e:
            print(f"⚠️  Skip {f.name}: {e}")
    print(f"✅  Loaded {len(episodes)} episodes")
    return episodes


def parse_timestamp(ep):
    ts_str = ep.get("timestamp", "")
    try:
        return datetime.fromisoformat(ts_str)
    except Exception:
        return None


# ── figure 1: relation type distribution ─────────────────────────────────────
def fig_relation_distribution(episodes):
    counts = Counter()
    for ep in episodes:
        rtypes = {r["relation_type"] for r in ep.get("relations", [])}
        if not rtypes:
            counts["none"] += 1
        for rt in rtypes:
            counts[rt] += 1

    labels = []
    values = []
    colors = []
    color_map = {
        "handover_ready": "#E74C3C",
        "tool_near":      "#3498DB",
        "none":           "#BDC3C7",
    }
    for k, c in [("handover_ready", color_map["handover_ready"]),
                 ("tool_near",      color_map["tool_near"]),
                 ("none",           color_map["none"])]:
        if counts[k] > 0:
            labels.append(k.replace("_", "\n"))
            values.append(counts[k])
            colors.append(c)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Number of Episodes", fontsize=12)
    ax.set_title("KG Relation Type Distribution\nacross 96 Episodes", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)

    plt.tight_layout()
    out = OUTPUT_DIR / "relation_type_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {out}")
    return counts


# ── figure 2: object co-occurrence heatmap ───────────────────────────────────
def fig_object_cooccurrence(episodes):
    # Count how many episodes contain a tool_near relation between (A, B)
    # Use unordered pairs (frozenset) so A-B and B-A are merged
    pair_counts = Counter()
    for ep in episodes:
        pairs_seen = set()
        for r in ep.get("relations", []):
            if r["relation_type"] == "tool_near":
                s, t = r["source_id"], r["target_id"]
                # Only count pairs where both are assembly objects
                if s in ASSEMBLY_OBJECTS and t in ASSEMBLY_OBJECTS and s != t:
                    pairs_seen.add(frozenset({s, t}))
        for pair in pairs_seen:
            pair_counts[tuple(sorted(pair))] += 1

    # Build symmetric matrix
    objs = ASSEMBLY_OBJECTS
    n = len(objs)
    idx = {o: i for i, o in enumerate(objs)}
    matrix = np.zeros((n, n), dtype=int)
    for (a, b), cnt in pair_counts.items():
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            matrix[i, j] = cnt
            matrix[j, i] = cnt

    fig, ax = plt.subplots(figsize=(8, 6.5))
    cmap = LinearSegmentedColormap.from_list("hrc", ["#EBF5FB", "#1A5276"])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [o.replace("screw_", "screw\n") for o in objs]
    ax.set_xticklabels(short, fontsize=9, rotation=35, ha="right")
    ax.set_yticklabels(short, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val > matrix.max() * 0.55 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Co-occurrence count (episodes)")
    ax.set_title("Object Co-occurrence in KG tool_near Relations\n(number of episodes where pair co-appeared)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "object_cooccurrence_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {out}")
    return pair_counts


# ── figure 3: KG activity timeline ───────────────────────────────────────────
def fig_activity_timeline(episodes):
    # Group episodes by date, separately tracking relation types
    dates_hr   = []  # dates with handover_ready
    dates_tn   = []  # dates with tool_near
    dates_none = []

    for ep in episodes:
        ts = parse_timestamp(ep)
        if ts is None:
            continue
        rtypes = {r["relation_type"] for r in ep.get("relations", [])}
        d = ts.date()
        if "handover_ready" in rtypes:
            dates_hr.append(d)
        if "tool_near" in rtypes:
            dates_tn.append(d)
        if not rtypes:
            dates_none.append(d)

    # Count per date
    from collections import Counter as C
    hr_c  = C(dates_hr)
    tn_c  = C(dates_tn)

    all_dates = sorted(set(list(hr_c.keys()) + list(tn_c.keys())))
    hr_vals  = [hr_c.get(d, 0) for d in all_dates]
    tn_vals  = [tn_c.get(d, 0) for d in all_dates]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(all_dates))
    width = 0.35
    ax.bar(x - width/2, hr_vals,  width, label="handover_ready", color="#E74C3C", alpha=0.85)
    ax.bar(x + width/2, tn_vals,  width, label="tool_near",      color="#3498DB", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(d)[5:] for d in all_dates], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Episode count", fontsize=11)
    ax.set_title("KG Activity Timeline\n(relation type per recording date)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "kg_activity_timeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {out}")


# ── figure 4: episode complexity distribution ─────────────────────────────────
def fig_episode_complexity(episodes):
    n_nodes = [len(ep.get("nodes", [])) for ep in episodes]
    n_rels  = [len(ep.get("relations", [])) for ep in episodes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ax1.hist(n_nodes, bins=range(min(n_nodes), max(n_nodes)+2), color="#2ECC71",
             edgecolor="white", linewidth=1.2)
    ax1.set_xlabel("Number of nodes per episode", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("KG Node Count\ndistribution", fontsize=12, fontweight="bold")
    ax1.axvline(np.median(n_nodes), color="red", linestyle="--", label=f"median={int(np.median(n_nodes))}")
    ax1.legend(fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Clip relation count for display (some episodes have many tool_near pairs)
    rels_clipped = np.clip(n_rels, 0, 70)
    ax2.hist(rels_clipped, bins=20, color="#9B59B6", edgecolor="white", linewidth=1.2)
    ax2.set_xlabel("Number of relations per episode\n(clipped at 70 for display)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("KG Relation Count\ndistribution", fontsize=12, fontweight="bold")
    ax2.axvline(np.median(n_rels), color="red", linestyle="--",
                label=f"median={int(np.median(n_rels))}")
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "episode_complexity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {out}")
    return n_nodes, n_rels


# ── print summary stats ───────────────────────────────────────────────────────
def print_stats(episodes, rel_counts, pair_counts, n_nodes, n_rels):
    print("\n" + "="*60)
    print("KG SNAPSHOT ANALYSIS — SUMMARY")
    print("="*60)
    print(f"Total episodes analyzed : {len(episodes)}")

    # date range
    tss = [parse_timestamp(ep) for ep in episodes]
    tss = [t for t in tss if t]
    if tss:
        print(f"Recording span          : {min(tss).date()} → {max(tss).date()}")
        days = (max(tss).date() - min(tss).date()).days
        print(f"                          ({days} days)")

    print(f"\nRelation type breakdown:")
    print(f"  handover_ready episodes : {rel_counts['handover_ready']}")
    print(f"  tool_near episodes      : {rel_counts['tool_near']}")
    print(f"  no-relation episodes    : {rel_counts['none']}")

    print(f"\nEpisode complexity:")
    print(f"  Nodes  — min:{min(n_nodes)}  median:{int(np.median(n_nodes))}  max:{max(n_nodes)}")
    print(f"  Rels   — min:{min(n_rels)}  median:{int(np.median(n_rels))}  max:{max(n_rels)}")

    print(f"\nTop-5 most common object co-occurrence pairs (tool_near):")
    for (a, b), cnt in pair_counts.most_common(5):
        print(f"  ({a}, {b}) : {cnt} episodes")

    print("\n" + "="*60)
    print(f"Figures saved to: {OUTPUT_DIR.resolve()}/")
    print("="*60)


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("📊 Analyzing KG snapshots...")
    episodes   = load_snapshots()

    print("\n[1/4] Relation type distribution...")
    rel_counts = fig_relation_distribution(episodes)

    print("[2/4] Object co-occurrence heatmap...")
    pair_counts = fig_object_cooccurrence(episodes)

    print("[3/4] KG activity timeline...")
    fig_activity_timeline(episodes)

    print("[4/4] Episode complexity distribution...")
    n_nodes, n_rels = fig_episode_complexity(episodes)

    print_stats(episodes, rel_counts, pair_counts, n_nodes, n_rels)
