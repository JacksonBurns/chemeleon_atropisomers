from pathlib import Path

import pandas as pd

dfs = []
data_dir = Path("cv_splits")
for repetition_number in range(5):
    subdir = data_dir / f"repetition_{repetition_number}"
    for fold_number in range(5):
        data_subdir = subdir / f"fold_{fold_number}"
        df = pd.read_csv(data_subdir / "test.csv")
        df["repetition_number"] = repetition_number
        df["chemeleon"] = pd.read_csv(data_subdir / "chemeleon_pred.csv")["ΔG(kcal/mol)"]
        df["physchem_forest"] = pd.read_csv(data_subdir / "physchem_forest_pred.csv")["pred"]
        df["unimol"] = pd.read_csv(data_subdir / "unimol_pred.csv")["pred"]
        dfs.append(df)

df = pd.concat(dfs)

# ai generated code to perform and visualize the stats tests for these results
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 1. Calculate Pearson R for each model per repetition
# We create a new summary table where each row is a (Repetition, Model, R-Value)
models = ['chemeleon', 'physchem_forest', 'unimol']
target = 'ΔG(kcal/mol)'
results_records = []

for rep in df['repetition_number'].unique():
    rep_df = df[df['repetition_number'] == rep]
    for model in models:
        # Calculate r for this specific CV fold
        r_val, _ = pearsonr(rep_df[target], rep_df[model])
        results_records.append({
            'Model': model, 
            'Pearson_R': r_val, 
            'Repetition': rep
        })

# Convert to a format Tukey HSD understands
stats_df = pd.DataFrame(results_records)

# 2. Run Tukey HSD Test
# endog: the metric we are comparing (Pearson R)
# groups: the categories we are comparing (Model names)
tukey = pairwise_tukeyhsd(endog=stats_df['Pearson_R'], 
                          groups=stats_df['Model'], 
                          alpha=0.05)

print(tukey)
print(stats_df.groupby("Model").mean()["Pearson_R"])
# 3. Visualization: Simultaneous Confidence Intervals
# Models with non-overlapping intervals are significantly different.
fig, ax = plt.subplots()
tukey.plot_simultaneous(ax=ax, comparison_name=stats_df.groupby("Model").mean()["Pearson_R"].idxmax(), figsize=(5, 3))

ax.set_title('Tukey HSD for Pearson $r$', fontsize=14)
ax.set_xlabel('Pearson $r$ Value', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
ax.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("tukey_hsd.png")

# ai generated visualization code
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

temp_col = 'T'

# keep only the first repetition for visualization purposes
df = df[df["repetition_number"] == 0]

# Calculate errors
for model in models:
    df[f'{model}_err'] = np.abs(df[model] - df[target])

fig, axes = plt.subplots(3, 2, figsize=(7*1.2, 9*1.2))
sns.set_theme(style="whitegrid")

for i, model in enumerate(models):
    # --- Parity Plot (Left Column) ---
    ax_p = axes[i, 0]
    sns.scatterplot(data=df, x=target, y=model, alpha=0.1, ax=ax_p, color='teal')
    
    # Metrics
    r2 = r2_score(df[target], df[model])
    mae = mean_absolute_error(df[target], df[model])
    rmse = np.sqrt(mean_squared_error(df[target], df[model]))
    
    # Identity line
    min_v, max_v = df[target].min(), df[target].max()
    ax_p.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2)
    
    stats_text = f"$R^2$: {r2:.3f}\nMAE: {mae:.3f}\nRMSE: {rmse:.3f}"
    ax_p.text(0.05, 0.95, stats_text, transform=ax_p.transAxes, verticalalignment='top', 
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_p.set_title(f'Parity: {model}', fontsize=14)
    ax_p.set_xlabel(f'Actual {target}')
    ax_p.set_ylabel(f'Predicted {target}')

    # --- Error vs Temp Plot (Right Column) ---
    ax_e = axes[i, 1]
    err_col = f'{model}_err'
    sns.scatterplot(data=df, x=temp_col, y=err_col, alpha=0.1, ax=ax_e, color='coral')
    sns.regplot(data=df, x=temp_col, y=err_col, scatter=False, ax=ax_e, color='black', line_kws={'linestyle':'--'})
    
    # Metrics
    corr, p = pearsonr(df[temp_col], df[err_col])
    mean_err = df[err_col].mean()
    err_stats = f"Corr (r): {corr:.3f} ({p=:.1e})"
    ax_e.text(0.05, 0.95, err_stats, transform=ax_e.transAxes, verticalalignment='top', 
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_e.set_title(f'Error vs Temp: {model}', fontsize=14)
    ax_e.set_xlabel('Temperature (K)')
    ax_e.set_ylabel('Absolute Error (kcal/mol)')

plt.tight_layout()
plt.savefig('parity.png', dpi=300)
