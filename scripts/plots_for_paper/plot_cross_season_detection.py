import matplotlib.pyplot as plt
import numpy as np

# Use a professional, academic style
plt.style.use('seaborn-v0_8-paper')

# Data organization
# Group 1: Evaluation on S2024 (In-season S24 vs Transfer from S25)
# Group 2: Evaluation on S2025 (In-season S25 vs Transfer from S24)
in_season_rates = [100.0, 92.0]
cross_season_no_adapt = [60.0, 95.0]

categories = ['Evaluation on S2024', 'Evaluation on S2025']
x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

# Creating the bars
rects1 = ax.bar(x - width/2, in_season_rates, width,
                label='In-Season',
                color='#2c3e50',
                edgecolor='black',
                alpha=0.9)

rects2 = ax.bar(x + width/2, cross_season_no_adapt, width,
                label='Cross-Season (No Adaptation)',
                color='#c0392b',
                edgecolor='black',
                alpha=0.9)

# Adding value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold',
                    fontsize=10)

autolabel(rects1)
autolabel(rects2)

# Styling and Titles
ax.set_title('Cross-Season Detection Asymmetry', fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel('Detection Rate (%)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylim(0, 115)

# Legend configuration
ax.legend(loc='upper right', frameon=True, fontsize=10)

# Visual refinements
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Visual separator between groups
ax.axvline(x=0.5, color='black', linestyle='-', linewidth=0.5, alpha=0.2)

plt.tight_layout()
plt.savefig("cross_season_asymmetry_final.png", dpi=300)
plt.show()