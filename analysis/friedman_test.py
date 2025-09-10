import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import matplotlib.pyplot as plt

# --- Load data ---
df = pd.read_excel("../Pipeline_Images/metric_tests/mastermetrictests.xlsx")

# --- Step 1: Calculate relative sharpness ---
# Define pixel count for each pipeline (p0 = raw ≈ 2400x4600, others = 600x600)
PIXELS = {
    "p0": 2400*4600,   # raw
    "p2": 600*600,
    "p5": 600*600,
    "p7": 600*600,
    "p8": 600*600,
    "p12": 600*600,
    "p13": 600*600,
}

df["rel_sharpness"] = df.apply(lambda row: row["sharpness"] / PIXELS.get(row["pipeline"], 600*600), axis=1)

# --- Step 2: Friedman test ---
wide = df.pivot(index="image_name", columns="pipeline", values="rel_sharpness").dropna()
k = wide.shape[1]
n = wide.shape[0]

stat, p = friedmanchisquare(*[wide[col] for col in wide.columns])
dfree = k - 1
kendalls_w = stat / (n * k * (k - 1))

# --- Step 3: Post-hoc Wilcoxon ---
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

# --- Step 4: Summary table ---
summary = pd.DataFrame([{
    "metric": "relative_sharpness",
    "friedman_chi2": stat,
    "df": dfree,
    "p_value": p,
    "kendalls_w": kendalls_w,
    "n_images": n,
    "k_pipelines": k
}])

# --- Step 5: Save results ---
with pd.ExcelWriter("friedman_test_sharpness.xlsx") as writer:
    summary.to_excel(writer, sheet_name="summary", index=False)
    posthoc.to_excel(writer, sheet_name="posthoc", index=False)

print("Saved friedman_test_sharpness.xlsx with summary + posthoc results.")

# --- Step 6: Plot boxplot ---
plt.figure(figsize=(8,6))
df.boxplot(column="rel_sharpness", by="pipeline", grid=False)
plt.title("Relative Sharpness by Pipeline")
plt.suptitle("")
plt.ylabel("Relative Sharpness")
plt.savefig("friedman_boxplot_sharpness.png", dpi=300, bbox_inches="tight")
plt.show()

