import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for saving inferences
if not os.path.exists('inferences'):
    os.makedirs('inferences')

# Load the dataset
df = pd.read_csv('titanic.csv')

# Read the patterns and anomalies file
with open('analysis/patterns_and_anomalies.txt', 'r') as f:
    patterns_text = f.read()

# Create a detailed feature-level inference document
with open('inferences/feature_inferences.md', 'w') as f:
    f.write("# Titanic Dataset: Feature-Level Inferences\n\n")
    
    f.write("## Introduction\n")
    f.write("This document presents detailed inferences for each feature in the Titanic dataset based on the exploratory data analysis (EDA) performed. These inferences are derived from summary statistics, visualizations, correlation analyses, and pattern identification.\n\n")
    
    # Passenger Class (Pclass)
    f.write("## Passenger Class (Pclass)\n\n")
    f.write("### Description\n")
    f.write("Pclass is a proxy for socio-economic status (SES): 1st = Upper, 2nd = Middle, 3rd = Lower\n\n")
    
    f.write("### Statistical Inferences\n")
    f.write("- Distribution: The dataset contains more 3rd class passengers (55.1%) than 1st (24.2%) or 2nd (20.7%) class\n")
    f.write("- Survival Correlation: Strong positive correlation with survival (higher class = higher survival chance)\n")
    f.write("- Survival Rates: 1st class (63.0%), 2nd class (47.3%), 3rd class (24.2%)\n\n")
    
    f.write("### Behavioral Inferences\n")
    f.write("- Class-based evacuation protocols likely favored higher classes\n")
    f.write("- 1st class cabins were likely closer to lifeboats and escape routes\n")
    f.write("- 1st class passengers may have received preferential treatment during evacuation\n")
    f.write("- 3rd class passengers may have had limited access to information about the emergency\n\n")
    
    f.write("### ML Implications\n")
    f.write("- Pclass is a strong predictor for survival models\n")
    f.write("- Should be retained in predictive models\n")
    f.write("- No need for feature engineering beyond potential one-hot encoding\n\n")
    
    # Sex
    f.write("## Sex\n\n")
    f.write("### Description\n")
    f.write("Passenger's gender (male/female)\n\n")
    
    f.write("### Statistical Inferences\n")
    f.write("- Distribution: Male passengers (64.8%) outnumber female passengers (35.2%)\n")
    f.write("- Survival Correlation: Strongest correlation with survival among all features\n")
    f.write("- Survival Rates: Females (74.2%), Males (18.9%)\n\n")
    
    f.write("### Behavioral Inferences\n")
    f.write("- Clear evidence of 'women and children first' evacuation policy\n")
    f.write("- Male passengers likely sacrificed their spots on lifeboats\n")
    f.write("- The extreme difference in survival rates suggests this was the primary factor in determining who was saved\n\n")
    
    f.write("### ML Implications\n")
    f.write("- Sex is the strongest predictor for survival models\n")
    f.write("- Essential feature for any predictive model\n")
    f.write("- Binary encoding is sufficient (0 for male, 1 for female)\n\n")
    
    # Age
    f.write("## Age\n\n")
    f.write("### Description\n")
    f.write("Passenger's age in years\n\n")
    
    f.write("### Statistical Inferences\n")
    f.write("- Distribution: Mean age of 29.7 years with standard deviation of 14.5 years\n")
    f.write("- Missing Values: 19.9% of age values are missing\n")
    f.write("- Outliers: Few outliers (1.23%) at the upper end (65-80 years)\n")
    f.write("- Survival Correlation: Moderate negative correlation with survival\n")
    f.write("- Survival Rates by Age Group: Children (58.0%), Teenagers (42.9%), Young Adults (38.3%), Adults (40.0%), Seniors (22.7%)\n\n")
    
    f.write("### Behavioral Inferences\n")
    f.write("- Children were prioritized during evacuation, especially young children\n")
    f.write("- Seniors had the lowest survival rate, possibly due to mobility issues or sacrificing their spots\n")
    f.write("- The relatively flat survival rate among teenagers, young adults, and adults suggests age was less important in these ranges\n\n")
    
    f.write("### ML Implications\n")
    f.write("- Age is a useful predictor but has missing values that require imputation\n")
    f.write("- Age binning (as done in our analysis) might be more effective than using raw age values\n")
    f.write("- Interaction effects between Age and Sex should be considered (e.g., 'child' status might override gender for evacuation priority)\n\n")
    
    # SibSp
    f.write("## SibSp (Siblings/Spouses)\n\n")
    f.write("### Description\n")
    f.write("Number of siblings or spouses aboard\n\n")
    
    f.write("### Statistical Inferences\n")
    f.write("- Distribution: Most passengers traveled without siblings/spouses (68.2%)\n")
    f.write("- Outliers: 5.16% of passengers had unusually large numbers of siblings/spouses (3-8)\n")
    f.write("- Survival Correlation: Weak negative correlation with survival\n\n")
    
    f.write("### Behavioral Inferences\n")
    f.write("- Passengers with 1-2 siblings/spouses had higher survival rates, suggesting small family units were advantageous\n")
    f.write("- Passengers with many siblings/spouses had lower survival rates, possibly due to difficulty staying together during evacuation\n")
    f.write("- Solo travelers had lower survival rates than those with 1-2 family members, suggesting some benefit to having close family support\n\n")
    
    f.write("### ML Implications\n")
    f.write("- More valuable when combined with Parch to create a 'FamilySize' feature\n")
    f.write("- Non-linear relationship with survival suggests binning or categorical transformation\n\n")
    
    # Parch
    f.write("## Parch (Parents/Children)\n\n")
    f.write("### Description\n")
    f.write("Number of parents or children aboard\n\n")
    
    f.write("### Statistical Inferences\n")
    f.write("- Distribution: Most passengers traveled without parents/children (76.1%)\n")
    f.write("- Outliers: 23.91% of values are considered outliers, suggesting unusual family structures\n")
    f.write("- Survival Correlation: Weak positive correlation with survival\n\n")
    
    f.write("### Behavioral Inferences\n")
    f.write("- Passengers with 1-3 parents/children had higher survival rates, consistent with the 'women and children first' policy\n")
    f.write("- Passengers with many parents/children had lower survival rates, suggesting difficulty in coordinating large family evacuations\n\n")
    
    f.write("### ML Implications\n")
    f.write("- More valuable when combined with SibSp to create a 'FamilySize' feature\n")
    f.write("- Consider creating a 'HasChildren' binary feature which might better capture the survival advantage\n\n")
    
    # Fare
    f.write("## Fare\n\n")
    f.write("### Description\n")
    f.write("Passenger fare (ticket price)\n\n")
    
    f.write("### Statistical Inferences\n")
    f.write("- Distribution: Highly right-skewed with mean of £32.2 and median of £14.5\n")
    f.write("- Outliers: 13.02% of values are outliers at the upper end (£66.6-£512.3)\n")
    f.write("- Survival Correlation: Moderate positive correlation with survival\n\n")
    
    f.write("### Behavioral Inferences\n")
    f.write("- Higher fare passengers had better survival rates, strongly correlated with passenger class\n")
    f.write("- The correlation between fare and survival is likely due to the advantages of higher-class accommodations (closer to lifeboats, better information)\n")
    f.write("- Extreme outliers in fare might represent luxury accommodations or large family bookings\n\n")
    
    f.write("### ML Implications\n")
    f.write("- Log transformation recommended due to high skewness\n")
    f.write("- High collinearity with Pclass suggests potential redundancy\n")
    f.write("- Fare per person (Fare divided by family size) might be more informative than raw fare\n\n")
    
    # Cabin
    f.write("## Cabin\n\n")
    f.write("### Description\n")
    f.write("Cabin number\n\n")
    
    f.write("### Statistical Inferences\n")
    f.write("- Missing Values: Extremely high missing rate (77.1%)\n")
    f.write("- Distribution: 147 unique cabin values among the 204 non-missing entries\n\n")
    
    f.write("### Behavioral Inferences\n")
    f.write("- Missing cabin information is likely correlated with lower class passengers\n")
    f.write("- Cabin letter (first character) indicates deck level, which correlates with proximity to lifeboats\n")
    f.write("- Passengers with recorded cabin numbers were more likely to survive, suggesting better accommodations or record-keeping for higher-class passengers\n\n")
    
    f.write("### ML Implications\n")
    f.write("- High missing rate makes this feature challenging to use directly\n")
    f.write("- Extracting cabin letter (deck) might provide value despite missing data\n")
    f.write("- 'HasCabin' binary feature (whether cabin information exists) could be a useful proxy for passenger status\n\n")
    
    # Embarked
    f.write("## Embarked\n\n")
    f.write("### Description\n")
    f.write("Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)\n\n")
    
    f.write("### Statistical Inferences\n")
    f.write("- Distribution: Southampton (72.4%), Cherbourg (18.9%), Queenstown (8.7%)\n")
    f.write("- Missing Values: Very few (0.22%, only 2 entries)\n")
    f.write("- Survival Correlation: Weak correlation with survival\n")
    f.write("- Survival Rates: Cherbourg (55.4%), Queenstown (39.0%), Southampton (33.7%)\n\n")
    
    f.write("### Behavioral Inferences\n")
    f.write("- Passengers embarking from Cherbourg had higher survival rates, possibly due to correlation with higher class passengers\n")
    f.write("- Southampton passengers had lower survival rates, possibly due to higher proportion of third-class passengers\n")
    f.write("- Port of embarkation might be a proxy for passenger wealth or nationality\n\n")
    
    f.write("### ML Implications\n")
    f.write("- One-hot encoding recommended for machine learning models\n")
    f.write("- Missing values can be imputed with the most common value (Southampton)\n")
    f.write("- Moderate predictive value, but likely less important than Sex, Pclass, or Age\n\n")
    
    # Ticket
    f.write("## Ticket\n\n")
    f.write("### Description\n")
    f.write("Ticket number\n\n")
    
    f.write("### Statistical Inferences\n")
    f.write("- Unique Values: 681 unique ticket numbers among 891 passengers\n")
    f.write("- Duplicate tickets likely represent family groups traveling together\n\n")
    
    f.write("### Behavioral Inferences\n")
    f.write("- Shared ticket numbers indicate passengers traveling together\n")
    f.write("- Ticket number format might contain information about booking class or agency\n\n")
    
    f.write("### ML Implications\n")
    f.write("- Raw ticket numbers have limited predictive value\n")
    f.write("- Feature engineering possibilities include extracting ticket prefixes or creating a 'GroupSize' feature based on shared tickets\n")
    f.write("- Generally less useful than other features without significant preprocessing\n\n")
    
    # Name
    f.write("## Name\n\n")
    f.write("### Description\n")
    f.write("Passenger name\n\n")
    
    f.write("### Statistical Inferences\n")
    f.write("- Unique Values: All 891 names are unique\n")
    f.write("- Contains titles (Mr, Mrs, Miss, etc.) that can be extracted\n\n")
    
    f.write("### Behavioral Inferences\n")
    f.write("- Titles extracted from names can indicate social status, age, and marital status\n")
    f.write("- Surnames can identify family groups\n\n")
    
    f.write("### ML Implications\n")
    f.write("- Raw names have no direct predictive value\n")
    f.write("- Extracted titles can be valuable predictors (e.g., 'Miss' vs 'Mrs' vs 'Mr')\n")
    f.write("- Surname extraction could help identify family groups when combined with fare and cabin information\n\n")
    
    # Combined Feature Inferences
    f.write("## Combined Feature Inferences\n\n")
    
    f.write("### Socio-Economic Status\n")
    f.write("- Pclass, Fare, and Cabin collectively represent socio-economic status\n")
    f.write("- Higher status passengers had significantly better survival chances\n")
    f.write("- This suggests preferential treatment during evacuation or better access to lifeboats\n\n")
    
    f.write("### Demographic Factors\n")
    f.write("- Sex and Age were critical survival determinants\n")
    f.write("- The 'women and children first' policy is strongly evident in the data\n")
    f.write("- Male passengers, especially adults, had significantly lower survival rates\n\n")
    
    f.write("### Family Structure\n")
    f.write("- Small families (3-4 total members) had optimal survival rates\n")
    f.write("- Very large families and solo travelers had lower survival rates\n")
    f.write("- This suggests both advantages of traveling with close family and disadvantages of coordinating large groups\n\n")
    
    f.write("### Location and Access\n")
    f.write("- Cabin location (implied by class and fare) likely affected access to lifeboats\n")
    f.write("- Higher-class accommodations were typically closer to the boat deck\n")
    f.write("- Third-class passengers may have had limited information about the emergency or restricted movement\n\n")
    
    f.write("## Conclusion\n\n")
    f.write("The Titanic dataset reveals clear patterns of survival based on demographic and socio-economic factors. Gender was the strongest predictor of survival, followed by passenger class. Age played a role, particularly for children, while family size showed a non-linear relationship with survival probability. These insights provide a foundation for both understanding the historical event and building predictive models.\n\n")
    
    f.write("The feature-level inferences suggest that a combination of social norms ('women and children first'), economic privilege (class-based access to lifeboats), and practical factors (cabin location, family coordination) determined survival outcomes during this maritime disaster.\n")

print("Feature-level inferences have been documented and saved to 'inferences/feature_inferences.md'")

# Create a summary visualization of feature importance for survival
plt.figure(figsize=(12, 8))
# Convert categorical variables to numeric for correlation analysis
df_encoded = df.copy()
df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})
df_encoded['Embarked'] = df_encoded['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_encoded['HasCabin'] = df_encoded['Cabin'].notna().astype(int)

# Calculate correlation with survival
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'HasCabin']
corr_with_survival = []
for feature in features:
    if feature in df_encoded.columns:
        correlation = df_encoded[['Survived', feature]].corr().iloc[0, 1]
        corr_with_survival.append((feature, abs(correlation)))

# Sort by absolute correlation
corr_with_survival.sort(key=lambda x: x[1], reverse=True)

# Plot
features = [x[0] for x in corr_with_survival]
correlations = [x[1] for x in corr_with_survival]
colors = ['#1f77b4' if corr >= 0 else '#d62728' for corr in correlations]

plt.figure(figsize=(12, 8))
bars = plt.barh(features, correlations, color=colors)
plt.xlabel('Absolute Correlation with Survival')
plt.title('Feature Importance for Survival Prediction')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('inferences/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Feature importance visualization has been saved to 'inferences/feature_importance.png'")
