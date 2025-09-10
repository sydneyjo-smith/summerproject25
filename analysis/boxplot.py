import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import os
import re

# =========================
# User inputs
# =========================
excel_path = r"/Users/sydneysmith/Desktop/Pipeline_Images/metric_tests/mastermetrictests.xlsx"  # Excel file
out_dir    = r"/Users/sydneysmith/Desktop/Pipeline_Images/figures_dissertation"                 # Save location
os.makedirs(out_dir, exist_ok=True)

# Preferred logical order (will auto-trim to only those present in the data)
PIPELINE_ORDER = ["p0", "p2", "p5", "p7", "p8", "p12", "p13"]

# =========================
# Styling: Times New Roman
# =========================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["figure.dpi"] = 100

# =========================
# Load
# =========================
df = pd.read_excel(excel_path)

# Basic checks
required_cols = {"image_name", "pipeline"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Identify metric columns (exclude id/keys and any you do not want like 'sharpness')
EXCLUDE = {"image_name", "pipeline"}
metrics = [c for c in df.columns if c not in EXCLUDE]

# Ensure pipeline is string like 'p0', 'p2', etc.
df["pipeline"] = df["pipeline"].astype(str)

# Keep only pipelines that match the expected pattern
df = df[df["pipeline"].str.match(r"^p\d+$", na=False)]

# Build order that actually exists in the sheet (respecting preferred order)
present = [p for p in PIPELINE_ORDER if p in df["pipeline"].unique()]
extra = sorted(
    [p for p in df["pipeline"].unique() if p not in present],
    key=lambda x: int(re.findall(r"\d+", x)[0])
)
ordered = present + extra

# Make categorical to control plotting order
df["pipeline"] = pd.Categorical(df["pipeline"], categories=ordered, ordered=True)

def pretty_label(s: str) -> str:
    """Turn metric_name into Title Case for axis/figure text."""
    return s.replace("_", " ").strip().title()

# Custom legend handles
legend_handles = [
    Line2D([0], [0], color="lightgray", lw=6, label="p0 = Baseline (Raw)"),
    Line2D([0], [0], color="#4C72B0", lw=6, label="p(n) = Pipeline n"),
]

# =========================
# Plot each metric
# =========================
for metric in metrics:
    if df[metric].dropna().empty:
        continue

    fig, ax = plt.subplots(figsize=(8, 6))

    # Define box colors: baseline gray, others blue
    box_colors = ["lightgray"] + ["#4C72B0"] * (len(ordered) - 1)

    # Collect data per pipeline
    data = [df.loc[df["pipeline"] == p, metric].dropna() for p in ordered]

    # Boxplot
    box = ax.boxplot(
        data,
        labels=ordered,
        patch_artist=True,
        widths=0.7,
        flierprops=dict(
            marker="o", markersize=3, linestyle="none",
            markerfacecolor="black", alpha=0.6
        ),
    )

    # Color boxes
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Labels and title
    ax.set_title(f"{pretty_label(metric)} Across Pre-processing Pipelines", fontsize=16, pad=20)
    ax.set_xlabel("Pipeline", fontsize=14)
    ax.set_ylabel(pretty_label(metric), fontsize=14)

    # Gridlines
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(False)

    # Legend outside
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{metric}_matplotlib.png"), dpi=300, bbox_inches="tight")
    plt.close()

# =========================
# Generate interpretive captions
# =========================
print("\nGenerated Captions:\n")

for metric in metrics:
    # Median per pipeline
    medians = df.groupby("pipeline")[metric].median().round(2)
    baseline = medians.get("p0", None)

    # Identify highest and lowest
    highest = medians.idxmax()
    lowest = medians.idxmin()

    # Variability check (range)
    variability = (df.groupby("pipeline")[metric]
                     .std()
                     .round(2)
                     .sort_values(ascending=False))
    most_var = variability.index[0]

    # Build interpretation
    interp = []
    if highest != "p0":
        interp.append(f"{highest} had the highest median")
    if lowest != "p0":
        interp.append(f"{lowest} had the lowest median")
    if most_var not in ["p0", highest]:
        interp.append(f"{most_var} showed the most variability")

    interpretation = "; ".join(interp) if interp else "Performance was broadly similar across pipelines"

    # Median summary
    median_summary = "; ".join([f"{p} = {m}" for p, m in medians.items()])

    # Full caption
    caption = (
        f"Figure: Distribution of {pretty_label(metric)} across preprocessing pipelines. "
        f"Medians were {median_summary}. "
        f"{interpretation}, relative to baseline (p0 = {baseline}). "
        f"Baseline is shown in gray, with all other pipelines in blue."
    )

    print(caption)
    print()
