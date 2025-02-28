import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from category_encoders import TargetEncoder

path = "Raw_Dataset_with_Issues.csv"
df = pd.read_csv(path)

# casting the age to int
#df['Age'] = df['Age'].astype('int32')


###############################   1   ############################
# removing rows with null values in any column
df.dropna(inplace=True) 

#############################   2  ################################
# # Identify duplicate rows
duplicates = df[df.duplicated()]
print("Duplicate rows:", duplicates)


#############################  3   ###########################
# Function to standardize categorical columns
def standardize_category(column, replacements):
    df[column] = df[column].str.lower().replace(replacements)
# Standardize 'City' column
city_replacements = {'new-york': 'new york', 'ny': 'new york'}
standardize_category('City', city_replacements)
# Standardize 'Gender' column
gender_replacements = {'new-york': 'new york', 'ny': 'new york'}
standardize_category('City', city_replacements)


###########################  4  ###########################
# IQR Method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    return data[(data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))]
outliers_iqr_salary = detect_outliers_iqr(df, 'Salary')
outliers_iqr_purchase = detect_outliers_iqr(df, 'Purchase_Amount')
# Z-score Method
outliers_zscore_salary = df[(np.abs(stats.zscore(df['Salary'])) > 3)]
outliers_zscore_purchase = df[(np.abs(stats.zscore(df['Purchase_Amount'])) > 3)]
# Compare the results
print("IQR Outliers in Salary:", outliers_iqr_salary)
print("Z-score Outliers in Salary:", outliers_zscore_salary)
print("IQR Outliers in Purchase_Amount:", outliers_iqr_purchase)
print("Z-score Outliers in Purchase_Amount:", outliers_zscore_purchase)



##########################  5  ################################
# casting the age to int 
df['Age'].astype('int32')
# Function to categorize age groups
def categorize_age(age):
    if 18 <= age <= 30:
        return 'Young'
    elif 31 <= age <= 50:
        return 'Middle-Aged'
    elif age >= 51:
        return 'Senior'
    else:
        return 'Unknown'
# Create new Age_Group column
df['Age_Group'] = df['Age'].apply(categorize_age)
# Display DataFrame
print(df)
# Visualization
sns.countplot(x='Age_Group', data=df, palette='viridis')
plt.title('Distribution of Customers by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()

##########################  6  ################################
# Histograms before normalization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Purchase_Amount'], kde=True, color='blue')
plt.title('Before Normalization (Original)')
# Min-Max Scaling
min_max_scaler = MinMaxScaler()
df['Purchase_Amount_MinMax'] = min_max_scaler.fit_transform(df[['Purchase_Amount']])
# Standardization (Z-score Normalization)
standard_scaler = StandardScaler()
df['Purchase_Amount_Standard'] = standard_scaler.fit_transform(df[['Purchase_Amount']])
# Histograms after normalization
plt.subplot(1, 2, 2)
sns.histplot(df['Purchase_Amount_MinMax'], kde=True, color='green', label='Min-Max')
sns.histplot(df['Purchase_Amount_Standard'], kde=True, color='red', label='Standard')
plt.title('After Normalization')
plt.legend()
plt.show()
# Display DataFrame
print(df)



##########################  7  ################################

# One-Hot Encoding
df_onehot = pd.get_dummies(df, columns=['City', 'Gender', 'Marital_Status', 'Education'])
 
# Label Encoding
label_encoder = LabelEncoder()
df_label_encoded = df.copy()
for column in ['City', 'Gender', 'Marital_Status', 'Education']:
    df_label_encoded[column] = label_encoder.fit_transform(df_label_encoded[column].astype(str))
 
df_target_encoded = df.copy()
for column in ['City', 'Gender', 'Marital_Status', 'Education']:
    target_mean = df.groupby(column)['Purchase_Amount'].mean()
    df_target_encoded[column] = df[column].map(target_mean)
 
print("\nOne-Hot Encoded DataFrame:")
print(df_onehot.head())
 
print("\nLabel Encoded DataFrame:")
print(df_label_encoded.head())
 
print("\nTarget Encoded DataFrame:")
print(df_target_encoded.head())
 

##########################  8  ################################
def correct_inconsistencies(df, column, mapping):
    df[column] = df[column].map(mapping)
    return df
 
marital_status_mapping = {
    'Single': 'Single',
    'Married': 'Married',
    'Divorced': 'Divorced',
    'Widowed': 'Widowed',
    'S': 'Single',
    'M': 'Married',
    'D': 'Divorced',
    'W': 'Widowed'
}
 
education_mapping = {
    'High School': 'High School',
    'Bachelor': 'Bachelor',
    'Master': 'Master',
    'PhD': 'PhD',
    'HS': 'High School',
    'BSc': 'Bachelor',
    'MSc': 'Master',
    'Doctorate': 'PhD'
}
 
df = correct_inconsistencies(df, 'Marital_Status', marital_status_mapping)
df = correct_inconsistencies(df, 'Education', education_mapping)
 
print(df[['Marital_Status', 'Education']].head())
##########################  9  ################################
##########################  10  ###############################

#creating a new file
new_path = 'cleanData.csv'
df.to_csv(new_path, index=False)

