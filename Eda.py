"""
F.Y. BCA Semester I (Div A) - Exploratory Data Analysis Project
Group No: 13 | Roll Nos: 80, 52, 54, 55, 57, 66, 75, 115
Dataset: UCI Iris Flower Analysis (bezdekIris.data & iris.data)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define Column Names from iris.names
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# 2. Load Both Extracted Data Files
df_bezdek = pd.read_csv('bezdekIris.data', header=None, names=columns)
df_original = pd.read_csv('iris.data', header=None, names=columns)

print("=== 1. AUDIT OF CHADWICK DISCREPANCY (iris.data vs bezdekIris.data) ===")
diff_mask = (df_original != df_bezdek).any(axis=1)
for idx in df_original[diff_mask].index:
    print(f"Sample #{idx + 1} (Index {idx}):")
    print(f"  iris.data (Uncorrected) : {df_original.loc[idx].to_dict()}")
    print(f"  bezdekIris.data (Fixed) : {df_bezdek.loc[idx].to_dict()}\n")

# Use the corrected Bezdek dataset for primary analysis
df = df_bezdek.copy()

print("=== 2. DATA QUALITY ASSESSMENT ===")
print(f"Dataset Shape (Rows, Columns): {df.shape}")
print(f"\nMissing Values per Column:\n{df.isnull().sum()}")
print(f"\nClass Distribution (Count & Percentage):")
class_counts = df['class'].value_counts()
class_pct = df['class'].value_counts(normalize=True) * 100
print(pd.DataFrame({'Count': class_counts, 'Percentage (%)': class_pct.round(1)}))

print("\n=== 3. DESCRIPTIVE STATISTICS ===")
desc_stats = df.describe().T[['count', 'min', 'max', 'mean', 'std']]
print(desc_stats.round(4))

print("\n=== 4. GROUPING AND AGGREGATION BY SPECIES ===")
grouped_means = df.groupby('class').mean()
grouped_min = df.groupby('class').min()
grouped_max = df.groupby('class').max()
print("\nSpecies-Wise Mean Values (cm):")
print(grouped_means.round(3))
print("\nSpecies-Wise Minimum Values (cm):")
print(grouped_min)
print("\nSpecies-Wise Maximum Values (cm):")
print(grouped_max)

print("\n=== 5. CORRELATION ANALYSIS ===")
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
print("Inter-Feature Pearson Correlation Matrix:")
print(corr_matrix.round(4))

# Calculate Class Correlation (Encoding Setosa=0, Versicolor=1, Virginica=2 as in iris.names)
df['class_code'] = df['class'].astype('category').cat.codes
class_corr = df.drop(columns=['class']).corr()['class_code'].drop('class_code')
print("\nFeature-to-Class Correlation (Matches iris.names):")
print(class_corr.round(4))

# === 6. VISUALIZATION GENERATION ===
sns.set_theme(style="whitegrid", palette="deep")

# Chart 1: Univariate Histograms & KDE
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
titles = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']

for ax, feature, title in zip(axes.flatten(), features, titles):
    sns.histplot(data=df, x=feature, hue='class', kde=True, element="step", ax=ax)
    ax.set_title(f'Univariate Distribution of {title}', fontweight='bold')

plt.tight_layout()
plt.savefig('1_univariate_distributions.png', dpi=300)
plt.close()

# Chart 2: Bivariate Scatter Plot (Petal Length vs Petal Width)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='class', style='class', s=90)
plt.axvline(x=2.45, color='red', linestyle='--', label='Setosa Linear Separation Boundary')
plt.title('Bivariate Analysis: Petal Length vs. Petal Width Clustering', fontweight='bold')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
plt.tight_layout()
plt.savefig('2_bivariate_clustering.png', dpi=300)
plt.close()

# Chart 3: Multivariate Pairplot
pairplot_fig = sns.pairplot(df.drop(columns=['class_code']), hue='class', diag_kind='kde', corner=True)
pairplot_fig.fig.suptitle('Multivariate Pairwise Relationships Across Iris Species', y=1.02, fontweight='bold')
pairplot_fig.savefig('3_multivariate_pairplot.png', dpi=300)
plt.close()

# Chart 4: Correlation Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".4f", cmap='Blues', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Pearson Correlation Heatmap (bezdekIris.data)', fontweight='bold')
plt.tight_layout()
plt.savefig('4_correlation_heatmap.png', dpi=300)
plt.close()

print("\nAll 4 high-resolution visualizations saved successfully!")
