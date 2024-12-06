import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df_togo=pd.read_csv('C:\\Users\\Abdilala\\Documents\\GitHub\\10_academy\\Data\\togo-dapaong_qc.csv')

# Summary statistics

print("print of summary togo")
summary_togo=df_togo.describe()
print(summary_togo)

# Check for missing values
missing_values_togo=df_togo.isnull().sum()
print(missing_values_togo)

# Check for outliers and incorrect entries
outliers_togo=df_togo[(df_togo['GHI']<0)| (df_togo['DNI']<0) | (df_togo['GHI']<0)]
print(outliers_togo)

#check for outliers sensor readings (ModA, ModB) and wind speed data (WS, WSgust) using IQR.
def outlier_iqr(data):
    Q1=data.quantile(0.25)
    Q3=data.quantile(0.75)
    IQR=Q3-Q1
    lb=Q1-1.5*IQR
    ub=Q3+1.5*IQR
    return(data<lb) | (data<ub)
#outlier for ModA, ModB, WS, WSgust
outliers_ModA = outlier_iqr(df_togo['ModA'])
outliers_ModB = outlier_iqr(df_togo['ModB'])
outliers_WS = outlier_iqr(df_togo['WS'])
outliers_WSgust =outlier_iqr(df_togo['WSgust'])

outliers = df_togo[outliers_ModA | outliers_ModB | outliers_WS | outliers_WSgust]
print("Detected outliers:")
print(outliers)

#we can visualize using matplotlib
#outliers are shown by Boxplot
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.boxplot(df_togo['ModA'])
plt.title('ModA Outliers')

plt.subplot(2, 2, 2)
plt.boxplot(df_togo['ModB'])
plt.title('ModB Outliers')

plt.subplot(2, 2, 3)
plt.boxplot(df_togo['WS'])
plt.title('WS Outliers')

plt.subplot(2, 2, 4)
plt.boxplot(df_togo['WSgust'])
plt.title('WSgust Outliers')

plt.tight_layout()
plt.show()
