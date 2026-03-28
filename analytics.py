import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data.csv")

# Set the visual style to match your previous plots
sns.set_theme(style="whitegrid")

plt.figure(figsize=(5, 5))

# Use scatterplot with alpha to handle overlapping points (200-1000 points)
sns.scatterplot(
    data=df, 
    x="T", 
    y="ΔG(kcal/mol)", 
    color="teal", 
    alpha=0.1, 
    edgecolor=None
)

# Add a trend line to show the general relationship between T and ΔG
sns.regplot(
    data=df, 
    x="T", 
    y="ΔG(kcal/mol)", 
    scatter=False, 
    color="darkslategrey", 
    line_kws={"linestyle": "--", "linewidth": 1.5}
)

# Refine labels and title
plt.title(r"Experimental $\Delta G$ vs. Temperature", fontsize=15)
plt.xlabel("Temperature (K)", fontsize=12)
plt.ylabel(r"$\Delta G$ (kcal/mol)", fontsize=12)

# Tighten layout and save/show
plt.tight_layout()
plt.savefig("analytics.png", dpi=300)