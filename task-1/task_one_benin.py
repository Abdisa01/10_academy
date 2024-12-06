import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df_benin = pd.read_csv('C:\\Users\\Abdilala\\Documents\\GitHub\\10_academy\\Data\\benin-malanville.csv')
print("print of summary Benin")
summary_benin = df_benin.describe()
print(summary_benin)


# Check for missing values
missing_values_benin = df_benin.isnull().sum()
print(missing_values_benin)

# Check for outliers and incorrect entries
outliers_benin = df_benin[(df_benin['GHI'] < 0) | (df_benin['DNI'] < 0) | (df_benin['DHI'] < 0)]
print(outliers_benin)

#check for outliers sensor readings (ModA, ModB) and wind speed data (WS, WSgust) using IQR.
def outlier_iqr(data):
    Q1=data.quantile(0.25)
    Q3=data.quantile(0.75)
    IQR=Q3-Q1
    lb=Q1-1.5*IQR
    ub=Q3+1.5*IQR
    return(data<lb) | (data<ub)
#outlier for ModA, ModB, WS, WSgust / there is also Z score method to detect outliears
outliers_ModA = outlier_iqr(df_benin['ModA'])
outliers_ModB = outlier_iqr(df_benin['ModB'])
outliers_WS = outlier_iqr(df_benin['WS'])
outliers_WSgust =outlier_iqr(df_benin['WSgust'])

outliers = df_benin[outliers_ModA | outliers_ModB | outliers_WS | outliers_WSgust]
print("Detected outliers:")
print(outliers)

#we can visualize using matplotlib
#outliers are shown by Boxplot
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.boxplot(df_benin['ModA'])
plt.title('ModA Outliers')

plt.subplot(2, 2, 2)
plt.boxplot(df_benin['ModB'])
plt.title('ModB Outliers')

plt.subplot(2, 2, 3)
plt.boxplot(df_benin['WS'])
plt.title('WS Outliers')

plt.subplot(2, 2, 4)
plt.boxplot(df_benin['WSgust'])
plt.title('WSgust Outliers')

plt.tight_layout()
plt.show()

# Convert date column to datetime
df_benin['Timestamp'] = pd.to_datetime(df_benin['Timestamp'])

# Plot GHI, DNI, DHI over time
plt.figure(figsize=(15, 7))
plt.plot(df_benin['Timestamp'], df_benin['GHI'], label='GHI')
plt.plot(df_benin['Timestamp'], df_benin['DNI'], label='DNI')
plt.plot(df_benin['Timestamp'], df_benin['DHI'], label='DHI')
plt.xlabel('Date')
plt.ylabel('Irradiance (W/m²)')
plt.title('Time Series of GHI, DNI, DHI')
plt.legend()
plt.show()

# Analyze the impact of cleaning on sensor readings
cleaned_data = df_benin[df_benin['Cleaning'] == 'Yes']
plt.scatter(cleaned_data['Timestamp'], cleaned_data['ModA'], label='ModA Cleaned')
plt.scatter(cleaned_data['Timestamp'], cleaned_data['ModB'], label='ModB Cleaned')
plt.title('Sensor Readings Over Time')
plt.legend()
plt.show()