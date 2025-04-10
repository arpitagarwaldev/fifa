import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read the data
df = pd.read_csv('artifacts/raw.csv')

# Convert value column to numeric, removing $ and formatting
def clean_value(val):
    val = str(val).replace('$', '').replace(',', '').replace(' ', '')
    if '.' in val:
        parts = val.split('.')
        if len(parts[-1]) > 2:  # If last part is more than 2 digits, it's probably thousands
            val = ''.join(parts)
    return float(val)

df['value'] = df['value'].apply(clean_value)

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# 1. Value Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['value'], bins=50)
plt.title('Distribution of Player Values')
plt.xlabel('Value ($)')
plt.ylabel('Count')
plt.savefig('plots/value_distribution.png')
plt.close()

# 2. Age vs Value
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='value')
plt.title('Age vs Player Value')
plt.xlabel('Age')
plt.ylabel('Value ($)')
plt.savefig('plots/age_vs_value.png')
plt.close()

# 3. Top 10 Most Valuable Countries
plt.figure(figsize=(12, 6))
df.groupby('country')['value'].mean().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Average Player Value by Country (Top 10)')
plt.xlabel('Country')
plt.ylabel('Average Value ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/top_countries.png')
plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(15, 12))
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Player Attributes')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png')
plt.close()

# Print some key statistics
print('\nKey Statistics:')
print(f'Total Players: {len(df)}')
print(f'Average Player Value: ${df["value"].mean():,.2f}')
print(f'Median Player Value: ${df["value"].median():,.2f}')
print(f'Most Valuable Player: {df.loc[df["value"].idxmax(), "player"]} (${df["value"].max():,.2f})')
print(f'Average Player Age: {df["age"].mean():.1f} years')
print(f'Number of Countries: {df["country"].nunique()}')
print(f'Number of Clubs: {df["club"].nunique()}')

# Print top correlations with value
print('\nTop 5 Attributes Most Correlated with Player Value:')
value_correlations = correlation['value'].sort_values(ascending=False)
print(value_correlations.head())
