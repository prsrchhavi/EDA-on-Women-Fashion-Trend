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
