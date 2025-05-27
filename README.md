# Titanic Dataset Exploratory Data Analysis (EDA)

This repository contains a comprehensive exploratory data analysis of the Titanic dataset, performed as part of an AI & ML internship task.

## Project Structure

```
titanic_eda/
├── data/
│   └── titanic.csv                  # Original Titanic dataset
├── scripts/
│   ├── eda.py                       # Initial data exploration and summary statistics
│   ├── visualizations.py            # Histograms and boxplots for numeric features
│   ├── feature_relationships.py     # Correlation analysis and feature relationships
│   ├── patterns_analysis.py         # Pattern and anomaly identification
│   └── feature_inferences.py        # Feature-level inferences
├── plots/
│   ├── histograms/                  # Distribution visualizations
│   ├── boxplots/                    # Boxplot visualizations
│   └── correlations/                # Correlation matrices and pairplots
├── analysis/                        # Analysis results and pattern identification
├── inferences/                      # Feature-level inferences and insights
├── titanic_eda_notebook.ipynb       # Jupyter notebook with complete analysis
├── interview_answers.md             # Answers to interview questions
├── summary_statistics.csv           # Summary statistics of the dataset
└── README.md                        # Project documentation
```

## Task Overview

This project focuses on Exploratory Data Analysis (EDA) of the Titanic dataset, with the following objectives:

1. Generate summary statistics (mean, median, std, etc.)
2. Create histograms and boxplots for numeric features
3. Use pairplot/correlation matrix for feature relationships
4. Identify patterns, trends, or anomalies in the data
5. Make feature-level inferences from visualizations

## Key Findings

### Survival Patterns

- **Gender**: Women had significantly higher survival rates (74.2%) than men (18.9%)
- **Class**: First class passengers had higher survival rates (63.0%) than second (47.3%) or third class (24.2%)
- **Age**: Children had higher survival rates than adults, with seniors having the lowest rates
- **Family Size**: Passengers with small families (1-3 members) had higher survival rates than solo travelers or those with large families

### Feature Relationships

- Strong correlation between passenger class and fare
- Moderate negative correlation between age and survival
- Strong correlation between gender and survival
- Non-linear relationship between family size and survival probability

### Data Quality Insights

- Missing values in Age (19.9%), Cabin (77.1%), and Embarked (0.2%)
- Outliers identified in Fare (13.0%), SibSp (5.2%), and Parch (23.9%)
- Highly skewed distribution of Fare values

## Interview Questions

The repository includes detailed answers to the following interview questions:

1. What is the purpose of EDA?
2. How do boxplots help in understanding a dataset?
3. What is correlation and why is it useful?
4. How do you detect skewness in data?
5. What is multicollinearity?
6. What tools do you use for EDA?
7. Can you explain a time when EDA helped you find a problem?
8. What is the role of visualization in ML?

## Tools Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations

## How to Run

1. Clone this repository
2. Install required packages: `pip install pandas numpy matplotlib seaborn plotly`
3. Run individual scripts or open the Jupyter notebook for the complete analysis

## Author

Created as part of the AI & ML Internship Task 2: Exploratory Data Analysis (EDA)
