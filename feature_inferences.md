# Titanic Dataset: Feature-Level Inferences

## Introduction
This document presents detailed inferences for each feature in the Titanic dataset based on the exploratory data analysis (EDA) performed. These inferences are derived from summary statistics, visualizations, correlation analyses, and pattern identification.

## Passenger Class (Pclass)

### Description
Pclass is a proxy for socio-economic status (SES): 1st = Upper, 2nd = Middle, 3rd = Lower

### Statistical Inferences
- Distribution: The dataset contains more 3rd class passengers (55.1%) than 1st (24.2%) or 2nd (20.7%) class
- Survival Correlation: Strong positive correlation with survival (higher class = higher survival chance)
- Survival Rates: 1st class (63.0%), 2nd class (47.3%), 3rd class (24.2%)

### Behavioral Inferences
- Class-based evacuation protocols likely favored higher classes
- 1st class cabins were likely closer to lifeboats and escape routes
- 1st class passengers may have received preferential treatment during evacuation
- 3rd class passengers may have had limited access to information about the emergency

### ML Implications
- Pclass is a strong predictor for survival models
- Should be retained in predictive models
- No need for feature engineering beyond potential one-hot encoding

## Sex

### Description
Passenger's gender (male/female)

### Statistical Inferences
- Distribution: Male passengers (64.8%) outnumber female passengers (35.2%)
- Survival Correlation: Strongest correlation with survival among all features
- Survival Rates: Females (74.2%), Males (18.9%)

### Behavioral Inferences
- Clear evidence of 'women and children first' evacuation policy
- Male passengers likely sacrificed their spots on lifeboats
- The extreme difference in survival rates suggests this was the primary factor in determining who was saved

### ML Implications
- Sex is the strongest predictor for survival models
- Essential feature for any predictive model
- Binary encoding is sufficient (0 for male, 1 for female)

## Age

### Description
Passenger's age in years

### Statistical Inferences
- Distribution: Mean age of 29.7 years with standard deviation of 14.5 years
- Missing Values: 19.9% of age values are missing
- Outliers: Few outliers (1.23%) at the upper end (65-80 years)
- Survival Correlation: Moderate negative correlation with survival
- Survival Rates by Age Group: Children (58.0%), Teenagers (42.9%), Young Adults (38.3%), Adults (40.0%), Seniors (22.7%)

### Behavioral Inferences
- Children were prioritized during evacuation, especially young children
- Seniors had the lowest survival rate, possibly due to mobility issues or sacrificing their spots
- The relatively flat survival rate among teenagers, young adults, and adults suggests age was less important in these ranges

### ML Implications
- Age is a useful predictor but has missing values that require imputation
- Age binning (as done in our analysis) might be more effective than using raw age values
- Interaction effects between Age and Sex should be considered (e.g., 'child' status might override gender for evacuation priority)

## SibSp (Siblings/Spouses)

### Description
Number of siblings or spouses aboard

### Statistical Inferences
- Distribution: Most passengers traveled without siblings/spouses (68.2%)
- Outliers: 5.16% of passengers had unusually large numbers of siblings/spouses (3-8)
- Survival Correlation: Weak negative correlation with survival

### Behavioral Inferences
- Passengers with 1-2 siblings/spouses had higher survival rates, suggesting small family units were advantageous
- Passengers with many siblings/spouses had lower survival rates, possibly due to difficulty staying together during evacuation
- Solo travelers had lower survival rates than those with 1-2 family members, suggesting some benefit to having close family support

### ML Implications
- More valuable when combined with Parch to create a 'FamilySize' feature
- Non-linear relationship with survival suggests binning or categorical transformation

## Parch (Parents/Children)

### Description
Number of parents or children aboard

### Statistical Inferences
- Distribution: Most passengers traveled without parents/children (76.1%)
- Outliers: 23.91% of values are considered outliers, suggesting unusual family structures
- Survival Correlation: Weak positive correlation with survival

### Behavioral Inferences
- Passengers with 1-3 parents/children had higher survival rates, consistent with the 'women and children first' policy
- Passengers with many parents/children had lower survival rates, suggesting difficulty in coordinating large family evacuations

### ML Implications
- More valuable when combined with SibSp to create a 'FamilySize' feature
- Consider creating a 'HasChildren' binary feature which might better capture the survival advantage

## Fare

### Description
Passenger fare (ticket price)

### Statistical Inferences
- Distribution: Highly right-skewed with mean of £32.2 and median of £14.5
- Outliers: 13.02% of values are outliers at the upper end (£66.6-£512.3)
- Survival Correlation: Moderate positive correlation with survival

### Behavioral Inferences
- Higher fare passengers had better survival rates, strongly correlated with passenger class
- The correlation between fare and survival is likely due to the advantages of higher-class accommodations (closer to lifeboats, better information)
- Extreme outliers in fare might represent luxury accommodations or large family bookings

### ML Implications
- Log transformation recommended due to high skewness
- High collinearity with Pclass suggests potential redundancy
- Fare per person (Fare divided by family size) might be more informative than raw fare

## Cabin

### Description
Cabin number

### Statistical Inferences
- Missing Values: Extremely high missing rate (77.1%)
- Distribution: 147 unique cabin values among the 204 non-missing entries

### Behavioral Inferences
- Missing cabin information is likely correlated with lower class passengers
- Cabin letter (first character) indicates deck level, which correlates with proximity to lifeboats
- Passengers with recorded cabin numbers were more likely to survive, suggesting better accommodations or record-keeping for higher-class passengers

### ML Implications
- High missing rate makes this feature challenging to use directly
- Extracting cabin letter (deck) might provide value despite missing data
- 'HasCabin' binary feature (whether cabin information exists) could be a useful proxy for passenger status

## Embarked

### Description
Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)

### Statistical Inferences
- Distribution: Southampton (72.4%), Cherbourg (18.9%), Queenstown (8.7%)
- Missing Values: Very few (0.22%, only 2 entries)
- Survival Correlation: Weak correlation with survival
- Survival Rates: Cherbourg (55.4%), Queenstown (39.0%), Southampton (33.7%)

### Behavioral Inferences
- Passengers embarking from Cherbourg had higher survival rates, possibly due to correlation with higher class passengers
- Southampton passengers had lower survival rates, possibly due to higher proportion of third-class passengers
- Port of embarkation might be a proxy for passenger wealth or nationality

### ML Implications
- One-hot encoding recommended for machine learning models
- Missing values can be imputed with the most common value (Southampton)
- Moderate predictive value, but likely less important than Sex, Pclass, or Age

## Ticket

### Description
Ticket number

### Statistical Inferences
- Unique Values: 681 unique ticket numbers among 891 passengers
- Duplicate tickets likely represent family groups traveling together

### Behavioral Inferences
- Shared ticket numbers indicate passengers traveling together
- Ticket number format might contain information about booking class or agency

### ML Implications
- Raw ticket numbers have limited predictive value
- Feature engineering possibilities include extracting ticket prefixes or creating a 'GroupSize' feature based on shared tickets
- Generally less useful than other features without significant preprocessing

## Name

### Description
Passenger name

### Statistical Inferences
- Unique Values: All 891 names are unique
- Contains titles (Mr, Mrs, Miss, etc.) that can be extracted

### Behavioral Inferences
- Titles extracted from names can indicate social status, age, and marital status
- Surnames can identify family groups

### ML Implications
- Raw names have no direct predictive value
- Extracted titles can be valuable predictors (e.g., 'Miss' vs 'Mrs' vs 'Mr')
- Surname extraction could help identify family groups when combined with fare and cabin information

## Combined Feature Inferences

### Socio-Economic Status
- Pclass, Fare, and Cabin collectively represent socio-economic status
- Higher status passengers had significantly better survival chances
- This suggests preferential treatment during evacuation or better access to lifeboats

### Demographic Factors
- Sex and Age were critical survival determinants
- The 'women and children first' policy is strongly evident in the data
- Male passengers, especially adults, had significantly lower survival rates

### Family Structure
- Small families (3-4 total members) had optimal survival rates
- Very large families and solo travelers had lower survival rates
- This suggests both advantages of traveling with close family and disadvantages of coordinating large groups

### Location and Access
- Cabin location (implied by class and fare) likely affected access to lifeboats
- Higher-class accommodations were typically closer to the boat deck
- Third-class passengers may have had limited information about the emergency or restricted movement

## Conclusion

The Titanic dataset reveals clear patterns of survival based on demographic and socio-economic factors. Gender was the strongest predictor of survival, followed by passenger class. Age played a role, particularly for children, while family size showed a non-linear relationship with survival probability. These insights provide a foundation for both understanding the historical event and building predictive models.

The feature-level inferences suggest that a combination of social norms ('women and children first'), economic privilege (class-based access to lifeboats), and practical factors (cabin location, family coordination) determined survival outcomes during this maritime disaster.
