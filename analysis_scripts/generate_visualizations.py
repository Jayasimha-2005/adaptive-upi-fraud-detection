import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# =============================================
# VISUALIZATION SCRIPT FOR ALL DATASETS
# =============================================

OUT_DIR = 'dataset_analysis/visualizations'
BAF_PATH = 'Datasets/BAF/Base.csv'
CC_PATH = 'Datasets/Credit card Fraud detection/credit_card_fraud_10k.csv'
PAYSIM_PATH = 'Datasets/Paysim/paysim dataset.csv'

# Load datasets
print("Loading datasets...")
df_baf = pd.read_csv(BAF_PATH)
df_cc = pd.read_csv(CC_PATH)
df_ps = pd.read_csv(PAYSIM_PATH, nrows=500000)  # sample for memory

print(f"BAF: {df_baf.shape}, CC: {df_cc.shape}, Paysim sample: {df_ps.shape}")

# ------------------------------------------------
# FIG 1: Dataset Size Comparison
# ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Dataset Size Comparison', fontsize=16, fontweight='bold', y=1.02)

datasets = ['BAF\n(Base)', 'PaySim', 'Credit Card\nFraud 10K']
rows = [1_000_000, 6_362_620, 10_000]
colors = ['#2196F3', '#4CAF50', '#FF9800']

ax = axes[0]
bars = ax.bar(datasets, rows, color=colors, edgecolor='black', linewidth=0.8)
ax.set_title('Total Rows per Dataset', fontweight='bold')
ax.set_ylabel('Number of Rows')
ax.set_yscale('log')
for bar, val in zip(bars, rows):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()*1.1,
            f'{val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# File sizes in MB
file_sizes = [203.6*6, 493.5, 0.36]  # BAF total (6 files), PaySim, CC
ax2 = axes[1]
bars2 = ax2.bar(datasets, file_sizes, color=colors, edgecolor='black', linewidth=0.8)
ax2.set_title('File Size on Disk (MB)', fontweight='bold')
ax2.set_ylabel('Size (MB)')
for bar, val in zip(bars2, file_sizes):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height()*1.02,
             f'{val:.0f} MB', ha='center', va='bottom', fontsize=10)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/01_dataset_size_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 1 saved.")

# ------------------------------------------------
# FIG 2: Fraud Distribution (all datasets)
# ------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Fraud Label Distribution Across Datasets', fontsize=16, fontweight='bold')

# BAF
baf_dist = df_baf['fraud_bool'].value_counts()
axes[0].pie([baf_dist.get(0,0), baf_dist.get(1,0)], 
            labels=['Legitimate\n(988,971)', 'Fraud\n(11,029)'],
            colors=['#4CAF50', '#F44336'], autopct='%1.2f%%', startangle=90,
            explode=(0, 0.08), textprops={'fontsize': 11})
axes[0].set_title('BAF Base\n(1,000,000 rows)', fontweight='bold')

# PaySim (full dataset stats)
ps_total = 6_362_620
ps_fraud = 8_213
axes[1].pie([ps_total - ps_fraud, ps_fraud],
            labels=[f'Legitimate\n({ps_total-ps_fraud:,})', f'Fraud\n({ps_fraud:,})'],
            colors=['#4CAF50', '#F44336'], autopct='%1.3f%%', startangle=90,
            explode=(0, 0.08), textprops={'fontsize': 11})
axes[1].set_title('PaySim\n(6,362,620 rows)', fontweight='bold')

# Credit Card
cc_dist = df_cc['is_fraud'].value_counts()
axes[2].pie([cc_dist.get(0,0), cc_dist.get(1,0)],
            labels=[f'Legitimate\n({cc_dist.get(0,0):,})', f'Fraud\n({cc_dist.get(1,0):,})'],
            colors=['#4CAF50', '#F44336'], autopct='%1.2f%%', startangle=90,
            explode=(0, 0.08), textprops={'fontsize': 11})
axes[2].set_title('Credit Card Fraud 10K\n(10,000 rows)', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/02_fraud_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2 saved.")

# ------------------------------------------------
# FIG 3: Transaction Amount Distribution
# ------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Transaction Amount Distributions', fontsize=16, fontweight='bold')

# BAF - intended_balcon_amount
ax = axes[0, 0]
baf_fraud_amt = df_baf[df_baf['fraud_bool']==1]['intended_balcon_amount']
baf_legit_amt = df_baf[df_baf['fraud_bool']==0]['intended_balcon_amount']
ax.hist(baf_legit_amt.clip(-20, 120), bins=80, alpha=0.6, color='#4CAF50', label='Legitimate', density=True)
ax.hist(baf_fraud_amt.clip(-20, 120), bins=80, alpha=0.6, color='#F44336', label='Fraud', density=True)
ax.set_title('BAF: intended_balcon_amount', fontweight='bold')
ax.set_xlabel('Amount (normalized)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(alpha=0.3)

# CC - amount
ax = axes[0, 1]
cc_fraud_amt = df_cc[df_cc['is_fraud']==1]['amount']
cc_legit_amt = df_cc[df_cc['is_fraud']==0]['amount']
ax.hist(cc_legit_amt.clip(0, 1000), bins=60, alpha=0.6, color='#4CAF50', label='Legitimate', density=True)
ax.hist(cc_fraud_amt.clip(0, 1000), bins=60, alpha=0.6, color='#F44336', label='Fraud', density=True)
ax.set_title('Credit Card: Amount Distribution', fontweight='bold')
ax.set_xlabel('Amount ($)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(alpha=0.3)

# PaySim - amount (sampled)
ax = axes[1, 0]
ps_fraud_sample = df_ps[df_ps['isFraud']==1]['amount']
ps_legit_sample = df_ps[df_ps['isFraud']==0]['amount'].sample(min(2000, len(df_ps[df_ps['isFraud']==0])), random_state=42)
ax.hist(np.log1p(ps_legit_sample), bins=60, alpha=0.6, color='#4CAF50', label='Legitimate', density=True)
if len(ps_fraud_sample) > 0:
    ax.hist(np.log1p(ps_fraud_sample), bins=60, alpha=0.6, color='#F44336', label='Fraud', density=True)
ax.set_title('PaySim: Amount Distribution (log1p)', fontweight='bold')
ax.set_xlabel('log1p(Amount)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(alpha=0.3)

# Box plot comparison
ax = axes[1, 1]
box_data = [
    np.log1p(df_cc[df_cc['is_fraud']==0]['amount']),
    np.log1p(df_cc[df_cc['is_fraud']==1]['amount']),
]
bp = ax.boxplot(box_data, tick_labels=['CC Legit', 'CC Fraud'], patch_artist=True,
                boxprops=dict(linewidth=1.5))
bp['boxes'][0].set_facecolor('#4CAF50')
bp['boxes'][1].set_facecolor('#F44336')
ax.set_title('CC Amount Box Plot (log scale)', fontweight='bold')
ax.set_ylabel('log1p(Amount)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/03_amount_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 saved.")

# ------------------------------------------------
# FIG 4: BAF Fraud Rate Over Time (Month)
# ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('BAF: Temporal Fraud Analysis', fontsize=16, fontweight='bold')

month_fraud_rate = df_baf.groupby('month')['fraud_bool'].mean() * 100
month_counts = df_baf.groupby('month').size()

ax = axes[0]
bars = ax.bar(month_fraud_rate.index, month_fraud_rate.values, color='#E91E63', edgecolor='black', linewidth=0.8)
ax.set_title('Fraud Rate by Month (%)', fontweight='bold')
ax.set_xlabel('Month (0=Month 0, 7=Month 7)')
ax.set_ylabel('Fraud Rate (%)')
ax.set_xticks(range(8))
for bar, val in zip(bars, month_fraud_rate.values):
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01, f'{val:.2f}%',
            ha='center', va='bottom', fontsize=9)
ax.grid(axis='y', alpha=0.3)

ax2 = axes[1]
ax2.bar(month_counts.index, month_counts.values, color='#2196F3', edgecolor='black', linewidth=0.8)
ax2.set_title('Transaction Volume by Month', fontweight='bold')
ax2.set_xlabel('Month')
ax2.set_ylabel('Transaction Count')
ax2.set_xticks(range(8))
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/04_baf_temporal_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4 saved.")

# ------------------------------------------------
# FIG 5: PaySim Fraud by Transaction Type
# ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('PaySim: Transaction Type Analysis', fontsize=16, fontweight='bold')

type_data = {
    'PAYMENT': {'total': 2_151_495, 'fraud': 0},
    'CASH_OUT': {'total': 2_237_500, 'fraud': 4_116},
    'CASH_IN': {'total': 1_399_284, 'fraud': 0},
    'TRANSFER': {'total': 532_909, 'fraud': 4_097},
    'DEBIT': {'total': 41_432, 'fraud': 0},
}

types = list(type_data.keys())
totals = [type_data[t]['total'] for t in types]
frauds = [type_data[t]['fraud'] for t in types]
fraud_rates = [type_data[t]['fraud']/type_data[t]['total']*100 for t in types]

x = np.arange(len(types))
width = 0.35
ax = axes[0]
bars1 = ax.bar(x - width/2, totals, width, label='Total', color='#2196F3', alpha=0.8)
bars2 = ax.bar(x + width/2, frauds, width, label='Fraud', color='#F44336', alpha=0.8)
ax.set_title('Transactions vs Fraud by Type', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(types, rotation=15)
ax.set_ylabel('Count')
ax.set_yscale('log')
ax.legend()
ax.grid(axis='y', alpha=0.3)

ax2 = axes[1]
colors_bar = ['#F44336' if r > 0 else '#9E9E9E' for r in fraud_rates]
ax2.bar(types, fraud_rates, color=colors_bar, edgecolor='black', linewidth=0.8)
ax2.set_title('Fraud Rate by Transaction Type (%)', fontweight='bold')
ax2.set_ylabel('Fraud Rate (%)')
ax2.set_xticklabels(types, rotation=15)
for i, (t, r) in enumerate(zip(types, fraud_rates)):
    ax2.text(i, r + 0.01, f'{r:.3f}%', ha='center', va='bottom', fontsize=10)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/05_paysim_type_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 5 saved.")

# ------------------------------------------------
# FIG 6: Missing Value Heatmap
# ------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle('Missing Values Analysis', fontsize=16, fontweight='bold')

for ax, (name, df, fraud_col) in zip(axes, [
    ('BAF Base', df_baf, 'fraud_bool'),
    ('Credit Card', df_cc, 'is_fraud'),
    ('PaySim (500K)', df_ps, 'isFraud')
]):
    miss_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    colors_miss = ['#F44336' if v > 10 else '#FF9800' if v > 5 else '#4CAF50' for v in miss_pct.values]
    ax.barh(range(len(miss_pct)), miss_pct.values, color=colors_miss)
    ax.set_yticks(range(len(miss_pct)))
    ax.set_yticklabels(miss_pct.index, fontsize=8)
    ax.set_xlabel('Missing %')
    ax.set_title(f'{name}\n({len(df.columns)} columns)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='black', linewidth=1)
    
    # Add annotation for all-zero case
    if miss_pct.max() == 0:
        ax.text(0.5, 0.5, 'NO MISSING VALUES\n✓ Complete Dataset', 
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/06_missing_values.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 6 saved.")

# ------------------------------------------------
# FIG 7: CC - Feature Analysis
# ------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Credit Card Dataset: Feature Analysis', fontsize=16, fontweight='bold')

cc_fraud = df_cc[df_cc['is_fraud']==1]
cc_legit = df_cc[df_cc['is_fraud']==0]

# Transaction hour
ax = axes[0, 0]
ax.hist(cc_legit['transaction_hour'], bins=24, alpha=0.6, color='#4CAF50', label='Legit', density=True)
ax.hist(cc_fraud['transaction_hour'], bins=24, alpha=0.6, color='#F44336', label='Fraud', density=True)
ax.set_title('Transaction Hour Distribution', fontweight='bold')
ax.set_xlabel('Hour of Day')
ax.legend()
ax.grid(alpha=0.3)

# Device trust score
ax = axes[0, 1]
ax.hist(cc_legit['device_trust_score'], bins=30, alpha=0.6, color='#4CAF50', label='Legit', density=True)
ax.hist(cc_fraud['device_trust_score'], bins=30, alpha=0.6, color='#F44336', label='Fraud', density=True)
ax.set_title('Device Trust Score Distribution', fontweight='bold')
ax.set_xlabel('Trust Score')
ax.legend()
ax.grid(alpha=0.3)

# Velocity last 24h
ax = axes[0, 2]
vel_fraud_rate = df_cc.groupby('velocity_last_24h')['is_fraud'].mean() * 100
ax.bar(vel_fraud_rate.index, vel_fraud_rate.values, color='#9C27B0', edgecolor='black')
ax.set_title('Fraud Rate by Velocity Last 24h', fontweight='bold')
ax.set_xlabel('Velocity (# transactions)')
ax.set_ylabel('Fraud Rate (%)')
ax.grid(axis='y', alpha=0.3)

# Merchant category
ax = axes[1, 0]
mc_fraud = df_cc.groupby('merchant_category')['is_fraud'].mean() * 100
mc_fraud.sort_values(ascending=True).plot(kind='barh', ax=ax, color='#FF5722')
ax.set_title('Fraud Rate by Merchant Category', fontweight='bold')
ax.set_xlabel('Fraud Rate (%)')
ax.grid(axis='x', alpha=0.3)

# Cardholder age
ax = axes[1, 1]
ax.hist(cc_legit['cardholder_age'], bins=20, alpha=0.6, color='#4CAF50', label='Legit', density=True)
ax.hist(cc_fraud['cardholder_age'], bins=20, alpha=0.6, color='#F44336', label='Fraud', density=True)
ax.set_title('Cardholder Age Distribution', fontweight='bold')
ax.set_xlabel('Age')
ax.legend()
ax.grid(alpha=0.3)

# Foreign transaction fraud rate
ax = axes[1, 2]
ft_data = df_cc.groupby('foreign_transaction')['is_fraud'].agg(['sum', 'count'])
ft_data['rate'] = ft_data['sum']/ft_data['count']*100
ax.bar(['Domestic\n(foreign=0)', 'Foreign\n(foreign=1)'], ft_data['rate'].values, 
       color=['#2196F3', '#FF9800'], edgecolor='black')
ax.set_title('Fraud Rate: Domestic vs Foreign', fontweight='bold')
ax.set_ylabel('Fraud Rate (%)')
for i, v in enumerate(ft_data['rate'].values):
    ax.text(i, v+0.05, f'{v:.2f}%', ha='center', fontsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/07_cc_feature_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 7 saved.")

# ------------------------------------------------
# FIG 8: BAF Feature Analysis
# ------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('BAF Dataset: Feature Analysis', fontsize=16, fontweight='bold')

baf_fraud = df_baf[df_baf['fraud_bool']==1]
baf_legit = df_baf[df_baf['fraud_bool']==0]

# Credit risk score
ax = axes[0, 0]
ax.hist(baf_legit['credit_risk_score'], bins=60, alpha=0.6, color='#4CAF50', label='Legit', density=True)
ax.hist(baf_fraud['credit_risk_score'], bins=60, alpha=0.6, color='#F44336', label='Fraud', density=True)
ax.set_title('Credit Risk Score Distribution', fontweight='bold')
ax.set_xlabel('Credit Risk Score')
ax.legend()
ax.grid(alpha=0.3)

# Housing status fraud rate
ax = axes[0, 1]
hs_fraud = df_baf.groupby('housing_status')['fraud_bool'].mean() * 100
hs_fraud.sort_values(ascending=True).plot(kind='barh', ax=ax, color='#E91E63')
ax.set_title('Fraud Rate by Housing Status', fontweight='bold')
ax.set_xlabel('Fraud Rate (%)')
ax.grid(axis='x', alpha=0.3)

# Device OS fraud rate
ax = axes[0, 2]
dos_fraud = df_baf.groupby('device_os')['fraud_bool'].mean() * 100
dos_fraud.sort_values(ascending=True).plot(kind='barh', ax=ax, color='#FF5722')
ax.set_title('Fraud Rate by Device OS', fontweight='bold')
ax.set_xlabel('Fraud Rate (%)')
ax.grid(axis='x', alpha=0.3)

# Session length
ax = axes[1, 0]
ax.hist(baf_legit['session_length_in_minutes'].clip(0, 60), bins=50, alpha=0.6, color='#4CAF50', label='Legit', density=True)
ax.hist(baf_fraud['session_length_in_minutes'].clip(0, 60), bins=50, alpha=0.6, color='#F44336', label='Fraud', density=True)
ax.set_title('Session Length Distribution', fontweight='bold')
ax.set_xlabel('Session Length (minutes)')
ax.legend()
ax.grid(alpha=0.3)

# Income distribution
ax = axes[1, 1]
income_fraud_rate = df_baf.groupby('income')['fraud_bool'].mean() * 100
ax.bar([str(round(x, 1)) for x in income_fraud_rate.index], income_fraud_rate.values, color='#9C27B0', edgecolor='black')
ax.set_title('Fraud Rate by Income Level', fontweight='bold')
ax.set_xlabel('Income (normalized 0.1-0.9)')
ax.set_ylabel('Fraud Rate (%)')
ax.grid(axis='y', alpha=0.3)

# Employment status fraud rate
ax = axes[1, 2]
emp_fraud = df_baf.groupby('employment_status')['fraud_bool'].mean() * 100
emp_fraud.sort_values(ascending=True).plot(kind='barh', ax=ax, color='#FF9800')
ax.set_title('Fraud Rate by Employment Status', fontweight='bold')
ax.set_xlabel('Fraud Rate (%)')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/08_baf_feature_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 8 saved.")

# ------------------------------------------------
# FIG 9: Dataset Comparison Radar Chart
# ------------------------------------------------
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(polar=True))

categories = ['Data Volume', 'Fraud Label\nQuality', 'Temporal\nInfo', 'LSTM/GRU\nSuitability',
              'Feature\nRichness', 'Behavioral\nFeatures', 'UPI\nRelevance', 'Concept Drift\nSuitability',
              'Streaming\nSuitability', 'Research\nReproducibility']

N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Scores (out of 10)
baf_scores = [8, 9, 5, 3, 8, 7, 3, 7, 4, 9]
paysim_scores = [9, 7, 8, 5, 5, 5, 3, 5, 7, 8]
cc_scores = [2, 5, 1, 1, 4, 3, 2, 1, 2, 4]

for scores, color, label in [(baf_scores, '#2196F3', 'BAF (Base)'),
                               (paysim_scores, '#4CAF50', 'PaySim'),
                               (cc_scores, '#FF9800', 'Credit Card 10K')]:
    values = scores + scores[:1]
    ax.plot(angles, values, 'o-', linewidth=2, color=color, label=label)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=10)
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels(['2', '4', '6', '8', '10'], size=8)
ax.set_title('Dataset Research Value Radar', size=16, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/09_dataset_radar_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 9 saved.")

# ------------------------------------------------
# FIG 10: PaySim - Step (temporal) distribution
# ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('PaySim: Temporal Distribution (Step = Hours)', fontsize=16, fontweight='bold')

# Fraud vs Legit by step (sampled)
ps_fraud_sample = df_ps[df_ps['isFraud']==1]
ps_legit_sample = df_ps[df_ps['isFraud']==0].sample(min(5000, len(df_ps[df_ps['isFraud']==0])), random_state=42)

ax = axes[0]
ax.hist(ps_legit_sample['step'], bins=50, alpha=0.6, color='#4CAF50', label='Legitimate', density=True)
if len(ps_fraud_sample) > 0:
    ax.hist(ps_fraud_sample['step'], bins=50, alpha=0.6, color='#F44336', label='Fraud', density=True)
ax.set_title('Transaction Step Distribution', fontweight='bold')
ax.set_xlabel('Step (Hour of Simulation, 1-743)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(alpha=0.3)
ax.text(0.05, 0.95, '743 hours ≈ 1 month', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5), fontsize=10)

# Type proportions
ax2 = axes[1]
type_data_plot = {'CASH_OUT': 2_237_500, 'PAYMENT': 2_151_495, 'CASH_IN': 1_399_284,
                  'TRANSFER': 532_909, 'DEBIT': 41_432}
colors_type = ['#F44336', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
wedges, texts, autotexts = ax2.pie(type_data_plot.values(), labels=type_data_plot.keys(),
                                     colors=colors_type, autopct='%1.1f%%', startangle=90)
ax2.set_title('Transaction Type Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/10_paysim_temporal.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 10 saved.")

# ------------------------------------------------
# FIG 11: Sequence Suitability Comparison
# ------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')

categories = ['Entity\nIdentifier', 'Temporal\nOrdering', 'Transaction\nHistory', 'Sequence\nLength', 
              'LSTM/GRU\nReady']

datasets_seq = ['BAF (Base)', 'PaySim', 'Credit Card 10K']
scores_seq = [
    [2, 3, 2, 2, 3],   # BAF - no user ID, month only, not sequential
    [7, 9, 5, 3, 5],   # PaySim - has nameOrig, step hours, but 99.99% single tx per user
    [3, 2, 1, 1, 2],   # CC - transaction_id only, hour only, no user history
]

x = np.arange(len(categories))
width = 0.25
colors_ds = ['#2196F3', '#4CAF50', '#FF9800']

for i, (ds, sc, col) in enumerate(zip(datasets_seq, scores_seq, colors_ds)):
    bars = ax.bar(x + i*width - width, sc, width, label=ds, color=col, alpha=0.85, edgecolor='black', linewidth=0.8)

ax.set_title('Sequence Modeling Suitability (Score 1-10)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel('Score (1=Poor, 10=Excellent)')
ax.set_ylim(0, 11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.4)
ax.axhline(y=7, color='green', linestyle='--', alpha=0.5, label='Good threshold')
ax.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='Poor threshold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/11_sequence_suitability.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 11 saved.")

print("\nAll visualizations saved to:", OUT_DIR)
