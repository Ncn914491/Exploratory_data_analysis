import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style for matplotlib plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Load the dataset
df = pd.read_csv('titanic.csv')

# Create a directory for saving plots
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# Display basic information about the dataset
print("Dataset Information:")
print(f"Shape: {df.shape}")
print("\nData Types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_data = pd.DataFrame({'Missing Values': missing_values, 
                            'Percentage': missing_percentage})
print(missing_data[missing_data['Missing Values'] > 0])

# Generate summary statistics
print("\nSummary Statistics:")
summary_stats = df.describe(include='all').T
summary_stats['missing'] = df.isnull().sum()
summary_stats['missing_percentage'] = (df.isnull().sum() / len(df)) * 100
print(summary_stats)

# Save summary statistics to CSV
summary_stats.to_csv('summary_statistics.csv')

print("\nEDA completed and summary statistics saved to 'summary_statistics.csv'")
