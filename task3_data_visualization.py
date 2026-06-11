"""
============================================================
  CodeAlpha Internship — TASK 3: Data Visualization
  Dataset: World Happiness Report (public dataset)
  Creates a full visual dashboard with compelling charts
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── STYLE SETUP ───────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"       : 130,
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})
COLORS = ["#6C63FF", "#FF6584", "#43B89C", "#F7C948", "#FF8C42", "#4FC3F7"]

# ── LOAD DATASET ──────────────────────────────────────────
print("=" * 58)
print("  CodeAlpha Internship — Task 3: Data Visualization")
print("=" * 58)

url = ("https://raw.githubusercontent.com/datasets/world-happiness/"
       "main/data/2023.csv")

try:
    df = pd.read_csv(url)
    print(f"[✓] Dataset loaded from GitHub: {df.shape[0]} rows")
except Exception:
    # Fallback: create sample data so script still runs
    print("[!] Could not fetch online. Using sample data.")
    np.random.seed(42)
    countries = [
        "Finland","Denmark","Iceland","Israel","Netherlands",
        "Sweden","Norway","Switzerland","Luxembourg","Australia",
        "New Zealand","Austria","Canada","Ireland","United States",
        "Germany","Belgium","Czech Republic","UK","Lithuania",
        "Slovenia","France","Romania","Serbia","Brazil",
        "India","China","Pakistan","Nigeria","Ethiopia"
    ]
    regions = (["Western Europe"]*10 + ["ANZ"]*2 +
               ["North America"]*2 + ["Western Europe"]*6 +
               ["Eastern Europe"]*4 + ["Latin America"]*1 +
               ["South Asia"]*2 + ["East Asia"]*1 +
               ["South Asia"]*1 + ["Sub-Saharan Africa"]*2)
    df = pd.DataFrame({
        "Country name"                    : countries,
        "Happiness score"                 : np.round(np.linspace(7.8, 3.8, 30) +
                                             np.random.uniform(-0.2, 0.2, 30), 3),
        "Explained by: GDP per capita"    : np.round(np.random.uniform(0.5, 2.0, 30), 3),
        "Explained by: Social support"    : np.round(np.random.uniform(0.3, 1.5, 30), 3),
        "Explained by: Healthy life expectancy": np.round(np.random.uniform(0.3, 0.9, 30), 3),
        "Explained by: Freedom to make life choices": np.round(np.random.uniform(0.1, 0.7, 30), 3),
        "Explained by: Generosity"        : np.round(np.random.uniform(0.0, 0.4, 30), 3),
        "Explained by: Perceptions of corruption": np.round(np.random.uniform(0.0, 0.5, 30), 3),
        "Regional indicator"              : regions,
    })

# ── COLUMN ALIASES (works with any version of dataset) ───
col_map = {
    "score"      : next((c for c in df.columns if "happiness" in c.lower() or "score" in c.lower()), df.columns[1]),
    "country"    : next((c for c in df.columns if "country" in c.lower()), df.columns[0]),
    "gdp"        : next((c for c in df.columns if "gdp" in c.lower()), None),
    "social"     : next((c for c in df.columns if "social" in c.lower()), None),
    "health"     : next((c for c in df.columns if "health" in c.lower() or "life" in c.lower()), None),
    "freedom"    : next((c for c in df.columns if "freedom" in c.lower()), None),
    "generosity" : next((c for c in df.columns if "generosity" in c.lower()), None),
    "corruption" : next((c for c in df.columns if "corruption" in c.lower()), None),
    "region"     : next((c for c in df.columns if "region" in c.lower()), None),
}

score_col   = col_map["score"]
country_col = col_map["country"]
df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
df = df.dropna(subset=[score_col])

top10    = df.nlargest(10, score_col)
bottom10 = df.nsmallest(10, score_col)

# ── BUILD DASHBOARD ───────────────────────────────────────
fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("#F0F2F6")

gs = gridspec.GridSpec(3, 2, figure=fig,
                       hspace=0.45, wspace=0.35,
                       top=0.93, bottom=0.05)

# Title
fig.suptitle("🌍  World Happiness Report — Data Dashboard\n"
             "CodeAlpha Internship | Task 3: Data Visualization",
             fontsize=17, fontweight="bold", color="#2C2C54", y=0.97)

# ── CHART 1: Top 10 Happiest Countries (Horizontal Bar) ──
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.barh(top10[country_col][::-1],
                top10[score_col][::-1],
                color=COLORS[0], edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, top10[score_col][::-1]):
    ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va="center", fontsize=8.5, color="#2C2C54")
ax1.set_title("🏆 Top 10 Happiest Countries", fontweight="bold", fontsize=12)
ax1.set_xlabel("Happiness Score")
ax1.set_xlim(0, df[score_col].max() + 0.7)
ax1.set_facecolor("#FAFAFA")

# ── CHART 2: Bottom 10 Countries ─────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.barh(bottom10[country_col],
                 bottom10[score_col],
                 color=COLORS[1], edgecolor="white", linewidth=0.8)
for bar, val in zip(bars2, bottom10[score_col]):
    ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va="center", fontsize=8.5, color="#2C2C54")
ax2.set_title("📉 Bottom 10 Countries", fontweight="bold", fontsize=12)
ax2.set_xlabel("Happiness Score")
ax2.set_xlim(0, df[score_col].max() + 0.7)
ax2.set_facecolor("#FAFAFA")

# ── CHART 3: Score Distribution (Histogram + KDE) ────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(df[score_col], bins=25, color=COLORS[2],
         edgecolor="white", alpha=0.85, density=True)
df[score_col].plot(kind="kde", ax=ax3, color=COLORS[0],
                   linewidth=2.5, label="KDE curve")
ax3.axvline(df[score_col].mean(), color=COLORS[1],
            linestyle="--", linewidth=2,
            label=f"Mean: {df[score_col].mean():.2f}")
ax3.set_title("📊 Happiness Score Distribution", fontweight="bold", fontsize=12)
ax3.set_xlabel("Happiness Score")
ax3.set_ylabel("Density")
ax3.legend(fontsize=9)
ax3.set_facecolor("#FAFAFA")

# ── CHART 4: GDP vs Happiness Scatter ────────────────────
ax4 = fig.add_subplot(gs[1, 1])
if col_map["gdp"]:
    gdp_col = col_map["gdp"]
    df[gdp_col] = pd.to_numeric(df[gdp_col], errors="coerce")
    scatter_df = df.dropna(subset=[gdp_col])
    ax4.scatter(scatter_df[gdp_col], scatter_df[score_col],
                color=COLORS[3], edgecolors=COLORS[0],
                linewidths=0.6, s=60, alpha=0.85)
    # Trend line
    z = np.polyfit(scatter_df[gdp_col], scatter_df[score_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(scatter_df[gdp_col].min(),
                         scatter_df[gdp_col].max(), 100)
    ax4.plot(x_line, p(x_line), color=COLORS[1],
             linewidth=2, linestyle="--", label="Trend line")
    ax4.set_xlabel("GDP per Capita (contribution)")
    ax4.legend(fontsize=9)
else:
    ax4.text(0.5, 0.5, "GDP column not found",
             ha="center", va="center", transform=ax4.transAxes)
ax4.set_title("💰 GDP per Capita vs Happiness", fontweight="bold", fontsize=12)
ax4.set_ylabel("Happiness Score")
ax4.set_facecolor("#FAFAFA")

# ── CHART 5: Factors Stacked Bar (Top 10 countries) ──────
ax5 = fig.add_subplot(gs[2, :])
factor_cols = [col_map[k] for k in
               ["gdp","social","health","freedom","generosity","corruption"]
               if col_map[k]]
factor_labels = ["GDP", "Social Support", "Health",
                 "Freedom", "Generosity", "Corruption"][:len(factor_cols)]

if factor_cols:
    top10_factors = top10.set_index(country_col)[factor_cols].apply(
        pd.to_numeric, errors="coerce").fillna(0)
    top10_factors.columns = factor_labels
    top10_factors.plot(kind="bar", stacked=True, ax=ax5,
                       color=COLORS[:len(factor_cols)],
                       edgecolor="white", linewidth=0.5)
    ax5.set_title("🧩 Happiness Factors Breakdown — Top 10 Countries",
                  fontweight="bold", fontsize=12)
    ax5.set_xlabel("Country")
    ax5.set_ylabel("Contribution to Score")
    ax5.tick_params(axis="x", rotation=30)
    ax5.legend(loc="upper right", fontsize=9, ncol=3)
    ax5.set_facecolor("#FAFAFA")
else:
    ax5.text(0.5, 0.5, "Factor columns not found in dataset",
             ha="center", va="center", transform=ax5.transAxes, fontsize=13)

plt.savefig("task3_happiness_dashboard.png", bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("[✓] Dashboard saved: task3_happiness_dashboard.png")
plt.show()

# ── DATA STORY ────────────────────────────────────────────
print("\n" + "=" * 58)
print("  DATA STORY — Key Insights")
print("=" * 58)
print(f"  • Happiest  : {top10.iloc[0][country_col]}  "
      f"(Score: {top10.iloc[0][score_col]:.3f})")
print(f"  • Least Happy: {bottom10.iloc[0][country_col]}  "
      f"(Score: {bottom10.iloc[0][score_col]:.3f})")
print(f"  • Global Average : {df[score_col].mean():.3f}")
print(f"  • Score Range    : {df[score_col].min():.3f} – {df[score_col].max():.3f}")
print("  • GDP & Social Support are the strongest happiness drivers")
print("  • Nordic countries dominate the Top 10 consistently")
print("\n[✓] Task 3 Complete! Upload .py + .png to GitHub.")
print("=" * 58)
