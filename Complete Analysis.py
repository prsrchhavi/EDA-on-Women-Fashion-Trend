# Women Fashion Dataset EDA 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency

# Set plot style
sns.set(style="whitegrid")


#Step 2: Load the Data
df = pd.read_csv(r"C:\Users\prsrc\Desktop\Python Project\summer-products.csv")  # Make sure the file name is correct

# Basic Overview
print("Dataset Shape:", df.shape)
print("\nInfo:")
print(df.info())

#Data Cleaning
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
df['rating'] = df['rating'].fillna(df['rating'].mean())

# Drop columns with too many missing values
df = df.drop(columns=['merchant_profile_picture'], errors='ignore')

# Converting data types if necessary
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("Data cleaned and saved!")

# Stats for selected numeric columns
numeric_cols = ['price', 'retail_price', 'units_sold', 'rating', 'rating_count', 'merchant_rating']

for col in numeric_cols:
    if col in df.columns:
        print(f"--- {col} ---")
        print(f"Mean: {df[col].mean():.2f}")
        print(f"Median: {df[col].median():.2f}")
        print(f"Mode: {df[col].mode().values}")
        print()


valid_data = df[(df['price'].notnull()) & (df['units_sold'] > 0)]
# Weighted mean price
weighted_mean_price = (valid_data['price'] * valid_data['units_sold']).sum() / valid_data['units_sold'].sum()
print("Weighted Mean Price (by Units Sold):", round(weighted_mean_price, 2))


# Objective 1: Price Analysis
df['discount'] = df['retail_price'] - df['price']

print("\nPrice Summary:")
print(df[['price', 'retail_price', 'discount']].describe())

plt.figure(figsize=(10,5))
sns.histplot(df['price'], kde=True, bins=50, color='skyblue')
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

#Objective 2: Rating Insights
print("\nRating Summary:")
print(df['rating'].describe())

plt.figure(figsize=(8,5))
sns.boxplot(x='uses_ad_boosts', y='rating', data=df)
plt.title("Rating vs Ad Boost")
plt.show()

#Objective 3: Sales Performance
top_products = df.sort_values(by='units_sold', ascending=False)[['title', 'units_sold']].head(10)
print("\nTop 10 Selling Products:")
print(top_products)

top_themes = df.groupby('theme')['units_sold'].sum().sort_values(ascending=False)

population_mean = 5000
sample_mean = df['units_sold'].mean()
sample_std = df['units_sold'].std()
sample_size = len(df)

# Z-test calculation
z_score = (sample_mean - population_mean) / (sample_std / np.sqrt(sample_size))
p_value = stats.norm.cdf(z_score)

print(f"Z-Score: {z_score}")
print(f"P-Value: {p_value}")

#Objective4: Badge distribution category through pie chart
df['badge_category'] = 'No badges'
for i in range(len(df)):
    row = df.iloc[i]  # Get current row
    local, quality, fast = row['badge_local_product'], row['badge_product_quality'], row['badge_fast_shipping']
    
    if local + quality + fast == 0:
        df.at[i, 'badge_category'] = 'No badges'
    elif local == 1 and quality == 0 and fast == 0:
        df.at[i, 'badge_category'] = 'Local only'
    elif quality == 1 and local == 0 and fast == 0:
        df.at[i, 'badge_category'] = 'Quality only'
    elif fast == 1 and local == 0 and quality == 0:
        df.at[i, 'badge_category'] = 'Fast only'
    elif local + quality == 2 and fast == 0:
        df.at[i, 'badge_category'] = 'Local + Quality'
    elif local + fast == 2 and quality == 0:
        df.at[i, 'badge_category'] = 'Local + Fast'
    elif quality + fast == 2 and local == 0:
        df.at[i, 'badge_category'] = 'Quality + Fast'
    elif local + quality + fast == 3:
        df.at[i, 'badge_category'] = 'All badges'


category_sales = df.groupby('badge_category')['units_sold'].sum().sort_values(ascending=False)# To calculate total units sold per category
colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2', '#073B4C', '#EF476F']
plt.figure(figsize=(10, 8))
wedges, _ = plt.pie(category_sales, colors=colors, startangle=90, wedgeprops={'width': 0.7})
plt.legend(
    wedges, 
    [f"{cat} ({sales} units, {sales/sum(category_sales):.1%})" for cat, sales in zip(category_sales.index, category_sales)],
    title="Badge Categories",
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)

plt.title('Units Sold by Badge Category', pad=20)
plt.tight_layout()
plt.show()

#Objective 5: Merchant Analysis
top_merchants = df.groupby('merchant_id')['merchant_rating'].mean().sort_values(ascending=False).head(10)
print("\nTop Rated Merchants:")
print(top_merchants)

plt.figure(figsize=(8,5))
sns.boxplot(x='merchant_has_profile_picture', y='merchant_rating', data=df)
plt.title("Merchant Profile Picture vs Rating")
plt.xlabel("Has Profile Picture (0 = No, 1 = Yes)")
plt.ylabel("Merchant Rating")
plt.show()

contingency_table = pd.crosstab(df['merchant_has_profile_picture'], df['theme'])

# Performing Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")

#Objective 6: Comparing Prices for every units sold over a period of time
grouped = df.groupby(df['price']).agg({
    'units_sold': 'mean',
    'rating_count': 'sum'
}).reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=grouped, x='price', y='units_sold', marker='o', label='Average Rating')
sns.lineplot(data=grouped, x='price', y='rating_count', marker='s', label='Total Ratings', color='orange')
plt.title('Price VS Units sold')
plt.xlabel('Price')
plt.ylabel('Units Sold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

# Objective 7: Correlation Heatmap
correlation_matrix = df[numeric_cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, linewidths=1)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()


# Objective 8: Merchant Rating Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['merchant_rating'], kde=True, bins=30, color='lightgreen')
plt.title("Merchant Rating Distribution")
plt.xlabel("Merchant Rating")
plt.ylabel("Count")
plt.show()

group_with_picture = df[df['merchant_has_profile_picture'] == 1]['merchant_rating']
group_without_picture = df[df['merchant_has_profile_picture'] == 0]['merchant_rating']

# Performing T-test
t_stat, p_value = stats.ttest_ind(group_with_picture, group_without_picture)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")



# Objective 9: Rating vs Units Sold
plt.figure(figsize=(10,6))
sns.scatterplot(x='rating', y='units_sold', data=df, color='purple')
plt.title("Rating vs Units Sold")
plt.xlabel("Rating")
plt.ylabel("Units Sold")
plt.show()


# Objective 10: Discount Analysis
plt.figure(figsize=(10,5))
sns.scatterplot(x='discount', y='units_sold', data=df, color='orange')
plt.title("Discount vs Units Sold")
plt.xlabel("Discount")
plt.ylabel("Units Sold")
plt.show()




