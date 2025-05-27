import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set style for matplotlib plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)

# Create directories for saving plots
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('plots/histograms'):
    os.makedirs('plots/histograms')
if not os.path.exists('plots/boxplots'):
    os.makedirs('plots/boxplots')

# Load the dataset
df = pd.read_csv('titanic.csv')

# List of numeric features for visualization
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']

# Create histograms for numeric features
print("Generating histograms for numeric features...")
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {feature}', fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'plots/histograms/{feature}_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a plotly version for interactive visualization
    fig = px.histogram(df, x=feature, marginal="box", title=f'Distribution of {feature}')
    fig.write_html(f'plots/histograms/{feature}_histogram_interactive.html')

# Create boxplots for numeric features
print("Generating boxplots for numeric features...")
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df[feature].dropna())
    plt.title(f'Boxplot of {feature}', fontsize=16)
    plt.ylabel(feature, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'plots/boxplots/{feature}_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create boxplots by survival status
print("Generating boxplots by survival status...")
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Survived', y=feature, data=df)
    plt.title(f'Boxplot of {feature} by Survival Status', fontsize=16)
    plt.xlabel('Survived (0=No, 1=Yes)', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'plots/boxplots/{feature}_by_survival_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create boxplots by passenger class
print("Generating boxplots by passenger class...")
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Pclass', y=feature, data=df)
    plt.title(f'Boxplot of {feature} by Passenger Class', fontsize=16)
    plt.xlabel('Passenger Class', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'plots/boxplots/{feature}_by_class_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create boxplots by gender
print("Generating boxplots by gender...")
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Sex', y=feature, data=df)
    plt.title(f'Boxplot of {feature} by Gender', fontsize=16)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'plots/boxplots/{feature}_by_gender_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create a combined figure showing distributions of all numeric features
plt.figure(figsize=(16, 12))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[feature].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {feature}', fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/combined_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a combined boxplot figure
plt.figure(figsize=(16, 12))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[feature].dropna())
    plt.title(f'Boxplot of {feature}', fontsize=14)
    plt.ylabel(feature, fontsize=12)
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/combined_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

print("All histograms and boxplots have been generated and saved to the 'plots' directory.")
