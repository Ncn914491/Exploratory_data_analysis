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

# Create directory for saving plots
if not os.path.exists('plots/correlations'):
    os.makedirs('plots/correlations')

# Load the dataset
df = pd.read_csv('titanic.csv')

# Convert categorical variables to numeric for correlation analysis
df_encoded = df.copy()
df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})
df_encoded['Embarked'] = df_encoded['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Select relevant features for correlation analysis
features_for_correlation = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
df_corr = df_encoded[features_for_correlation].dropna()

# Create correlation matrix
print("Generating correlation matrix...")
plt.figure(figsize=(12, 10))
corr_matrix = df_corr.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            linewidths=0.5, cbar_kws={'shrink': .8})
plt.title('Correlation Matrix of Titanic Dataset Features', fontsize=16)
plt.tight_layout()
plt.savefig('plots/correlations/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Create an interactive correlation matrix with plotly
fig = px.imshow(corr_matrix, 
                text_auto=True, 
                color_continuous_scale='RdBu_r',
                title='Interactive Correlation Matrix')
fig.write_html('plots/correlations/correlation_matrix_interactive.html')

# Create pairplot for numeric features
print("Generating pairplot...")
plt.figure(figsize=(16, 14))
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
pairplot = sns.pairplot(df, vars=numeric_features, hue='Survived', palette='viridis', 
                        diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Pairplot of Numeric Features by Survival Status', y=1.02, fontsize=16)
pairplot.savefig('plots/correlations/pairplot.png', dpi=300, bbox_inches='tight')
plt.close()

# Create pairplot with passenger class
print("Generating pairplot by passenger class...")
pairplot_class = sns.pairplot(df, vars=numeric_features, hue='Pclass', palette='viridis', 
                              diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Pairplot of Numeric Features by Passenger Class', y=1.02, fontsize=16)
pairplot_class.savefig('plots/correlations/pairplot_by_class.png', dpi=300, bbox_inches='tight')
plt.close()

# Create pairplot with gender
print("Generating pairplot by gender...")
pairplot_sex = sns.pairplot(df, vars=numeric_features, hue='Sex', palette='viridis', 
                            diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Pairplot of Numeric Features by Gender', y=1.02, fontsize=16)
pairplot_sex.savefig('plots/correlations/pairplot_by_gender.png', dpi=300, bbox_inches='tight')
plt.close()

# Create scatter plots for key relationships
print("Generating scatter plots for key relationships...")
# Age vs Fare with survival coloring
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, palette='viridis', alpha=0.7)
plt.title('Age vs Fare by Survival Status', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Fare', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('plots/correlations/age_vs_fare_by_survival.png', dpi=300, bbox_inches='tight')
plt.close()

# Interactive scatter plot with plotly
fig = px.scatter(df, x='Age', y='Fare', color='Survived', 
                 size='Fare', hover_data=['Pclass', 'Sex', 'SibSp', 'Parch'],
                 title='Interactive Scatter Plot: Age vs Fare by Survival Status')
fig.write_html('plots/correlations/age_vs_fare_interactive.html')

# Create a correlation heatmap focused on survival
plt.figure(figsize=(10, 8))
survival_corr = corr_matrix['Survived'].sort_values(ascending=False)
sns.heatmap(pd.DataFrame(survival_corr), annot=True, fmt='.2f', cmap='coolwarm', 
            linewidths=0.5, cbar_kws={'shrink': .8})
plt.title('Correlation with Survival', fontsize=16)
plt.tight_layout()
plt.savefig('plots/correlations/survival_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

print("All correlation analyses and pairplots have been generated and saved to the 'plots/correlations' directory.")
