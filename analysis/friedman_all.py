import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import matplotlib.pyplot as plt
import os

# --- Load data ---
df = pd.read_excel("../Pipeline_Images/metric_tests/mastermetrictests.xlsx")

# --- Metrics to analyze (exclude sharpness) ---
metrics = [c for c in df.columns if c not in ["image_name", "pipeline", "sharpness"]]

# --- Output folder ---
out_dir = "../Pipeline_Images"
os.makedirs(out_dir, exist_ok=True)

summary_rows = []
posthoc_results = {}

for metric in metrics:
    print(f"Running Friedman test for {metric}...")

    # pivot to wide
    wide = df.pivot(index="image_name", columns="pipeline", values=metric).dropna()
    k = wide.shape[1]
    n = wide.shape[0]

    if k < 3:
        print(f"Skipping {metric}: not enough pipelines")
        continue

    # Friedman test
    stat, p = friedmanchisquare(*[wide[col] for col in wide.columns])
    dfree = k - 1
    kendalls_w = stat / (n * k * (k - 1))

    # Post-hoc Wilcoxon
    pairs = list(combinations(wide.columns, 2))
    pvals = []
    for a, b in pairs:
        res = wilcoxon(wide[a], wide[b])
        pvals.append(res.pvalue)

    reject, pvals_adj, _, _ = multipletests(pvals, method="holm")
    posthoc = pd.DataFrame({
        "comparison": [f"{a} vs {b}" for a,b in pairs],
        "p_raw": pvals,
        "p_holm": pvals_adj,
        "significant": reject
    }).sort_values("p_holm")

    posthoc_results[metric] = posthoc

    # Add to summary
    summary_rows.append({
        "metric": metric,
        "friedman_chi2": stat,
        "df": dfree,
        "p_value": p,
        "kendalls_w": kendalls_w,
        "n_images": n,
        "k_pipelines": k
    })

    # Plot
    plt.figure(figsize=(8,6))
    df.boxplot(column=metric, by="pipeline", grid=False)
    plt.title(f"{metric} by Pipeline")
    plt.suptitle("")
    plt.ylabel(metric)
    plt.savefig(os.path.join(out_dir, f"{metric}.png"), dpi=300, bbox_inches="tight")
    plt.close()

# --- Save all results into one Excel ---
summary = pd.DataFrame(summary_rows)
excel_path = os.path.join(out_dir, "friedman_all_metrics.xlsx")

with pd.ExcelWriter(excel_path) as writer:
    summary.to_excel(writer, sheet_name="summary", index=False)
    for metric, posthoc in posthoc_results.items():
        posthoc.to_excel(writer, sheet_name=f"posthoc_{metric}", index=False)

print(f"All results saved to {excel_path}")
print(f"Plots saved to {out_dir}")
