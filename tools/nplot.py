#!python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# -----------------------------
# CONFIG
# -----------------------------
METRIC = "unigram_entropy"
METRICS2 = "BOUNDARY_F1"
# METRIC = "TTR"
# METRIC = "bigram_conditional_entropy"

# -----------------------------
# Language-level feature dict
# ----------------------------
bylang_dict = {"ces":{'unigram_entropy': 6.880776890317845, 'bigram_conditional_entropy': 0.8980047488526365, 'num_tokens': 379, 'num_types': 196, 'TTR': 0.5171503957783641},
"deu": {'unigram_entropy': 6.491820656870884, 'bigram_conditional_entropy': 0.9604813091302106, 'num_tokens': 366, 'num_types': 174, 'TTR': 0.47540983606557374},
"ell": {'unigram_entropy': 7.42034761133714, 'bigram_conditional_entropy': 0.5059233058294078, 'num_tokens': 351, 'num_types': 225, 'TTR': 0.6410256410256411},
"eng": {'unigram_entropy': 7.126227313030544, 'bigram_conditional_entropy': 0.043728373050213785, 'num_tokens': 213, 'num_types': 172, 'TTR': 0.8075117370892019},
"epo": {'unigram_entropy': 6.201398139107223, 'bigram_conditional_entropy': 0.2918335145615976, 'num_tokens': 353, 'num_types': 177, 'TTR': 0.5014164305949008},
"fra": {'unigram_entropy': 6.945132852549051, 'bigram_conditional_entropy': 0.29160641827572203, 'num_tokens': 305, 'num_types': 194, 'TTR': 0.6360655737704918},
"hin": {'unigram_entropy': 6.987872692502161, 'bigram_conditional_entropy': 0.11646357762350808, 'num_tokens': 204, 'num_types': 156, 'TTR': 0.7647058823529411},
"hrv": {'unigram_entropy': 6.260447973840434, 'bigram_conditional_entropy': 0.8936518128627231, 'num_tokens': 397, 'num_types': 177, 'TTR': 0.44584382871536526},
"ita": {'unigram_entropy': 6.157624301505331, 'bigram_conditional_entropy': 0.5765747034213482, 'num_tokens': 353, 'num_types': 177, 'TTR': 0.5014164305949008},
"jpn": {'unigram_entropy': 7.663149936241201, 'bigram_conditional_entropy': 0.1490392676986209, 'num_tokens': 249, 'num_types': 219, 'TTR': 0.8795180722891566},
"lat": {'unigram_entropy': 6.935486780511901, 'bigram_conditional_entropy': 0.6788231181599913, 'num_tokens': 377, 'num_types': 205, 'TTR': 0.5437665782493368},
"mar": {'unigram_entropy': 7.197656347791997, 'bigram_conditional_entropy': 0.7841408164613654, 'num_tokens': 221, 'num_types': 175, 'TTR': 0.7918552036199095},
"pol": {'unigram_entropy': 6.599900818203124, 'bigram_conditional_entropy': 1.154559478660857, 'num_tokens': 414, 'num_types': 176, 'TTR': 0.4251207729468599},
"rus": {'unigram_entropy': 6.780833744546667, 'bigram_conditional_entropy': 1.143755653925216, 'num_tokens': 433, 'num_types': 201, 'TTR': 0.46420323325635104},
"spa": {'unigram_entropy': 7.045951238725733, 'bigram_conditional_entropy': 0.3426663235760658, 'num_tokens': 321, 'num_types': 207, 'TTR': 0.6448598130841121},
"tel": {'unigram_entropy': 7.5109768426568095, 'bigram_conditional_entropy': 0.1268824275675865, 'num_tokens': 219, 'num_types': 194, 'TTR': 0.8858447488584474},
"ukr": {'unigram_entropy': 6.930830793155453, 'bigram_conditional_entropy': 0.9545944168536822, 'num_tokens': 398, 'num_types': 198, 'TTR': 0.49748743718592964},
"zho": {'unigram_entropy': 7.7497441018114985, 'bigram_conditional_entropy': 0.28957334844199817, 'num_tokens': 292, 'num_types': 235, 'TTR': 0.8047945205479452}}


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(sys.argv[1], sep="\t")
df["LANGUAGE"] = df["LANGUAGE"].str.replace("results_3/", "")

# -----------------------------
# Aggregate
# -----------------------------
df_plot = df.groupby(["LANGUAGE", "LLM"])[METRICS2].mean().reset_index()

# map metric
df_plot[METRIC] = df_plot["LANGUAGE"].map(
    {lang: vals.get(METRIC, None) for lang, vals in bylang_dict.items()}
)

df_plot = df_plot.dropna(subset=[METRIC, METRICS2])

# -----------------------------
# Correlation (overall)
# -----------------------------
x = df_plot[METRIC].values
y = df_plot[METRICS2].values

pearson_corr, pearson_p = pearsonr(x, y)
spearman_corr, spearman_p = spearmanr(x, y)

print("\n=== OVERALL CORRELATION ===")
print(f"Pearson r  : {pearson_corr:.4f} (p={pearson_p:.4g})")
print(f"Spearman ρ : {spearman_corr:.4f} (p={spearman_p:.4g})")

# -----------------------------
# Correlation per LLM
# -----------------------------
print("\n=== PER-LLM CORRELATION ===")

for llm, sub in df_plot.groupby("LLM"):
    if len(sub) < 3:
        print(f"{llm}: not enough data")
        continue

    r, p = pearsonr(sub[METRIC], sub[METRICS2])
    rho, p2 = spearmanr(sub[METRIC], sub[METRICS2])

    print(f"{llm}: Pearson r={r:.4f}, p={p:.4f}, Spearman ρ={rho:.4f}, p={p2:.4f}")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 6))

for llm, sub in df_plot.groupby("LLM"):
    plt.scatter(sub[METRIC], sub[METRICS2], label=llm)

    for _, row in sub.iterrows():
        plt.text(
            row[METRIC],
            row[METRICS2],
            row["LANGUAGE"],
            fontsize=8,
            ha='right',
            va='bottom'
        )

plt.xlabel(f"Morphological Complexity ({METRIC})")
plt.ylabel("Boundary F1")
plt.title(f"{METRIC} vs LLM Performance (colored by LLM)")
plt.legend(title="LLM")

plt.tight_layout()
plt.savefig(f"scatter_{METRIC}_f1_llm.jpg", dpi=300)
plt.show()
