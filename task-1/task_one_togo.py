import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df_togo=pd.read_csv('C:\\Users\\Lalaa\\Documents\\GitHub\\10_academy\\Data\\togo-dapaong_qc.csv')

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

# Convert date column to datetime
df_togo['Timestamp'] = pd.to_datetime(df_togo['Timestamp'])

# Plot GHI, DNI, DHI over time
plt.figure(figsize=(15, 7))
plt.plot(df_togo['Timestamp'], df_togo['GHI'], label='GHI')
plt.plot(df_togo['Timestamp'], df_togo['DNI'], label='DNI')
plt.plot(df_togo['Timestamp'], df_togo['DHI'], label='DHI')
plt.xlabel('Date')
plt.ylabel('Irradiance (W/m²)')
plt.title('Time Series of GHI, DNI, DHI')
plt.legend()
plt.show()

# Analyze the impact of cleaning on sensor readings
cleaned_data = df_togo[df_togo['Cleaning'] == 'Yes']
plt.scatter(cleaned_data['Timestamp'], cleaned_data['ModA'], label='ModA Cleaned')
plt.scatter(cleaned_data['Timestamp'], cleaned_data['ModB'], label='ModB Cleaned')
plt.title('Sensor Readings Over Time')
plt.legend()
plt.show()

#commulative GHI and DNI visualization 
df_togo['Timestamp']= pd.to_datetime(df_togo['Timestamp'])
df_togo.set_index('Timestamp', inplace=True)
df_togo['Cumulative_GHI']=df_togo['GHI'].cumsum()
df_togo['Cumulative_DNI']=df_togo['DNI'].cumsum()
plt.figure(figsize=(14, 8))
plt.plot(df_togo.index, df_togo['Cumulative_GHI'], label='Cumulative GHI', color='orange')
plt.plot(df_togo.index, df_togo['Cumulative_DNI'], label='Cumulative DNI', color='blue')
plt.title('Cumulative GHI and DNI Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Irradiance (Wh/m²)')
plt.legend()
plt.grid()
plt.show()

print("Available columns in data:", df_togo.columns.tolist())
#relevant column to correlete 
rel_column=['GHI', 'DNI', 'DHI', 'TModA', 'TModB', 'WS', 'WSgust', 'WD']
data= df_togo[rel_column].dropna() #to drop nan values
  
#correlation between SOlar radiation and Temperature Measures
corr_matrix=data[['GHI', 'DNI', 'DHI', 'TModA', 'TModB']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Matrix_ solar Radiation and Tempureture Measures')
plt.show()

# Pair Plot for Solar Radiation and Temperature
sns.pairplot(data[['GHI', 'DNI', 'DHI', 'TModA', 'TModB']])
plt.suptitle('Pair Plot: Solar Radiation and Temperature Measures', y=1.02)
plt.show()

# Scatter Matrix for Wind Conditions and Solar Irradiance
pd.plotting.scatter_matrix(data[['WS', 'WSgust', 'WD', 'GHI', 'DNI', 'DHI']], figsize=(12, 12), diagonal='kde')
plt.suptitle('Scatter Matrix: Wind Conditions and Solar Irradiance', y=1.02)
plt.show()

data=df_togo[['WS','WD' ]].dropna()
data['WD_rad']=np.radians(data['WD'])
def wind_rose(data, num_bins=8): 
    # Create bins for wind direction
    wind_bins = np.linspace(0, 360, num_bins + 1)
    bin_labels = [(wind_bins[i], wind_bins[i + 1]) for i in range(num_bins)]
    
    # Create a DataFrame to hold wind speed averages per direction bin
    wind_rose_data = pd.Series(pd.cut(data['WD'], wind_bins)).value_counts().sort_index()
    
    # Create radial plot
    angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False).tolist()
    speeds = wind_rose_data.values / wind_rose_data.sum()  # Normalize counts to get probabilities

    # Close the plot
    speeds = np.concatenate((speeds, [speeds[0]]))
    angles += angles[:1]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, speeds, color='blue', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{int((label[0] + label[1]) / 2)}°" for label in bin_labels])
    ax.set_title('Wind Rose Plot', fontsize=16)
    plt.show()

# Call the function to create a wind rose
wind_rose(data)
plt.figure(figsize=(10, 6))
sns.histplot(data['WS'], bins=30, kde=True, color='teal')
plt.title('Wind Speed Distribution')
plt.xlabel('Wind Speed (WS)')
plt.ylabel('Frequency')
plt.show()

# Analyze wind direction variability
plt.figure(figsize=(10, 6))
sns.histplot(data['WD'], bins=36, kde=True, color='orange')
plt.title('Wind Direction Distribution')
plt.xlabel('Wind Direction (Degrees)')
plt.ylabel('Frequency')
plt.show()

# Temperature Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['RH'], y=data['TModA'], color='orange')
plt.title('Relative Humidity vs Temperature (TModA)')
plt.xlabel('Relative Humidity (RH)')
plt.ylabel('Temperature (TModA)')
plt.show()

# 8. Histograms
data[['GHI', 'DNI', 'DHI', 'WS', 'Tamb']].hist(bins=30, figsize=(15, 10))
plt.suptitle('Histograms of Key Variables')
plt.show()

# 9. Z-Score Analysis
data['z_score_GHI'] = (data['GHI'] - data['GHI'].mean()) / data['GHI'].std()
outlier_gli = data[data['z_score_GHI'].abs() > 3]
print("\nOutliers in GHI based on Z-score:")
print(outlier_gli[['Date', 'GHI']])

# 10. Bubble Chart
plt.figure(figsize=(10, 6))
plt.scatter(data['GHI'], data['Tamb'], s=data['RH']*10, alpha=0.5, c='blue', edgecolors='w')
plt.title('Bubble Chart: GHI vs Tamb (Size = RH)')
plt.xlabel('GHI')
plt.ylabel('Tamb')
plt.show()

# 11. Data Cleaning
# Handle anomalies and missing values
data_cleaned = data.dropna(subset=['Comments'])  # Adjust as needed
data_cleaned = data_cleaned[data_cleaned['Comments'] != '']  # Remove empty comments
print("\nCleaned Data:")
print(data_cleaned.head())