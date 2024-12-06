import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df_sierreleone=pd.read_csv('C:\\Users\\Abdilala\\Documents\\GitHub\\10_academy\\Data\\sierraleone-bumbuna.csv')


# Summary statistics

print("print of summary sierreleone")
summary_sierrelone=df_sierreleone.describe()
print(summary_sierrelone)

# Check for missing values
missing_values_sierrelone=df_sierreleone.isnull().sum()
print(missing_values_sierrelone)

# Check for outliers and incorrect entries
outliers_sierralone=df_sierreleone[(df_sierreleone['GHI']<0)|(df_sierreleone["DNI"]<0)|(df_sierreleone['DHI']<0)]
print(outliers_sierralone)

#check for outliers sensor readings (ModA, ModB) and wind speed data (WS, WSgust) using IQR.
def outlier_iqr(data):
    Q1=data.quantile(0.25)
    Q3=data.quantile(0.75)
    IQR=Q3-Q1
    lb=Q1-1.5*IQR
    ub=Q3+1.5*IQR
    return(data<lb) | (data<ub)
#outlier for ModA, ModB, WS, WSgust
outliers_ModA = outlier_iqr(df_sierreleone['ModA'])
outliers_ModB = outlier_iqr(df_sierreleone['ModB'])
outliers_WS = outlier_iqr(df_sierreleone['WS'])
outliers_WSgust =outlier_iqr(df_sierreleone['WSgust'])

outliers = df_sierreleone[outliers_ModA | outliers_ModB | outliers_WS | outliers_WSgust]
print("Detected outliers:")
print(outliers)

#we can visualize using matplotlib
#outliers are shown by Boxplot
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.boxplot(df_sierreleone['ModA'])
plt.title('ModA Outliers')

plt.subplot(2, 2, 2)
plt.boxplot(df_sierreleone['ModB'])
plt.title('ModB Outliers')

plt.subplot(2, 2, 3)
plt.boxplot(df_sierreleone['WS'])
plt.title('WS Outliers')

plt.subplot(2, 2, 4)
plt.boxplot(df_sierreleone['WSgust'])
plt.title('WSgust Outliers')

plt.tight_layout()
plt.show()
