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

# Create directory for saving analysis
if not os.path.exists('analysis'):
    os.makedirs('analysis')

# Load the dataset
df = pd.read_csv('titanic.csv')

# Convert categorical variables to numeric for analysis
df_encoded = df.copy()
df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})
df_encoded['Embarked'] = df_encoded['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Analyze survival rates by different features
print("Analyzing survival patterns...")

# Overall survival rate
overall_survival = df['Survived'].mean() * 100
print(f"Overall survival rate: {overall_survival:.2f}%")

# Survival by gender
survival_by_gender = df.groupby('Sex')['Survived'].mean() * 100
print("\nSurvival rate by gender:")
print(survival_by_gender)

# Survival by class
survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
print("\nSurvival rate by passenger class:")
print(survival_by_class)

# Survival by age groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                        labels=['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior'])
survival_by_age = df.groupby('AgeGroup')['Survived'].mean() * 100
print("\nSurvival rate by age group:")
print(survival_by_age)

# Survival by embarkation point
survival_by_embarked = df.groupby('Embarked')['Survived'].mean() * 100
print("\nSurvival rate by embarkation point:")
print(survival_by_embarked)

# Survival by family size (SibSp + Parch)
df['FamilySize'] = df['SibSp'] + df['Parch']
survival_by_family = df.groupby('FamilySize')['Survived'].mean() * 100
print("\nSurvival rate by family size:")
print(survival_by_family)

# Identify outliers in numeric features
print("\nIdentifying outliers in numeric features...")
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']

for feature in numeric_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    print(f"\nOutliers in {feature}:")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Percentage of outliers: {len(outliers) / len(df) * 100:.2f}%")
    print(f"Range of outliers: {outliers[feature].min()} to {outliers[feature].max()}")

# Create visualizations for identified patterns

# Survival by gender visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='Sex', y='Survived', data=df, palette='viridis')
plt.title('Survival Rate by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('analysis/survival_by_gender.png', dpi=300, bbox_inches='tight')
plt.close()

# Survival by class visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', data=df, palette='viridis')
plt.title('Survival Rate by Passenger Class', fontsize=16)
plt.xlabel('Passenger Class', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('analysis/survival_by_class.png', dpi=300, bbox_inches='tight')
plt.close()

# Survival by age group visualization
plt.figure(figsize=(12, 6))
sns.barplot(x='AgeGroup', y='Survived', data=df, palette='viridis')
plt.title('Survival Rate by Age Group', fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('analysis/survival_by_age.png', dpi=300, bbox_inches='tight')
plt.close()

# Survival by embarkation point visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='Embarked', y='Survived', data=df, palette='viridis')
plt.title('Survival Rate by Embarkation Point', fontsize=16)
plt.xlabel('Embarkation Point (C=Cherbourg, Q=Queenstown, S=Southampton)', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('analysis/survival_by_embarked.png', dpi=300, bbox_inches='tight')
plt.close()

# Survival by family size visualization
plt.figure(figsize=(12, 6))
family_survival = df.groupby('FamilySize')['Survived'].mean().reset_index()
sns.barplot(x='FamilySize', y='Survived', data=family_survival, palette='viridis')
plt.title('Survival Rate by Family Size', fontsize=16)
plt.xlabel('Family Size (SibSp + Parch)', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('analysis/survival_by_family_size.png', dpi=300, bbox_inches='tight')
plt.close()

# Class and gender combined effect on survival
plt.figure(figsize=(12, 6))
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df, palette='viridis')
plt.title('Survival Rate by Class and Gender', fontsize=16)
plt.xlabel('Passenger Class', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('analysis/survival_by_class_and_gender.png', dpi=300, bbox_inches='tight')
plt.close()

# Interactive visualization of survival patterns
fig = px.sunburst(df, path=['Sex', 'Pclass', 'Survived'], 
                  color='Survived', 
                  color_continuous_scale='viridis',
                  title='Hierarchical View of Survival Patterns')
fig.write_html('analysis/survival_patterns_sunburst.html')

# Save analysis results to file
with open('analysis/patterns_and_anomalies.txt', 'w') as f:
    f.write("# Titanic Dataset: Patterns, Trends, and Anomalies\n\n")
    
    f.write("## Overall Statistics\n")
    f.write(f"Overall survival rate: {overall_survival:.2f}%\n\n")
    
    f.write("## Survival Patterns\n")
    f.write("### Survival by Gender\n")
    f.write(f"{survival_by_gender.to_string()}\n")
    f.write("- Women had a significantly higher survival rate than men\n\n")
    
    f.write("### Survival by Passenger Class\n")
    f.write(f"{survival_by_class.to_string()}\n")
    f.write("- First class passengers had the highest survival rate\n")
    f.write("- Third class passengers had the lowest survival rate\n\n")
    
    f.write("### Survival by Age Group\n")
    f.write(f"{survival_by_age.to_string()}\n")
    f.write("- Children had higher survival rates\n")
    f.write("- Seniors had lower survival rates\n\n")
    
    f.write("### Survival by Embarkation Point\n")
    f.write(f"{survival_by_embarked.to_string()}\n")
    f.write("- Passengers who embarked from Cherbourg (C) had higher survival rates\n")
    f.write("- Passengers who embarked from Southampton (S) had lower survival rates\n\n")
    
    f.write("### Survival by Family Size\n")
    f.write(f"{survival_by_family.to_string()}\n")
    f.write("- Passengers with small families (1-3 members) had higher survival rates\n")
    f.write("- Passengers traveling alone or with very large families had lower survival rates\n\n")
    
    f.write("## Outliers and Anomalies\n")
    for feature in numeric_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        f.write(f"### Outliers in {feature}\n")
        f.write(f"- Number of outliers: {len(outliers)}\n")
        f.write(f"- Percentage of outliers: {len(outliers) / len(df) * 100:.2f}%\n")
        f.write(f"- Range of outliers: {outliers[feature].min()} to {outliers[feature].max()}\n\n")
    
    f.write("## Key Trends and Patterns\n")
    f.write("1. Gender was the strongest predictor of survival, with women having much higher survival rates\n")
    f.write("2. Social class (indicated by ticket class) strongly influenced survival chances\n")
    f.write("3. Age played a role in survival, with children having priority\n")
    f.write("4. Family size affected survival, with small families having better chances\n")
    f.write("5. The port of embarkation correlated with survival rates\n")
    f.write("6. There is a significant interaction effect between gender and class\n")

print("Analysis of patterns, trends, and anomalies completed and saved to 'analysis/patterns_and_anomalies.txt'")
