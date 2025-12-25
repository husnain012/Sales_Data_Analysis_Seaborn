import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("="*50)
print("SALES DATA ANALYSIS")
print("="*50)
print("\n")

# 1) Load and explore data
data = pd.read_csv("sales_data.csv")
print("--------------- Overview ---------------")
print("\n")
print(data.head())
print("\n")
print(data.info())
print("\n")

# 2) Total sales by product category
total_sales_category = data.groupby("product_category")["sales_amount"].sum()
print("--------------- Total sales by product category ---------------")
print("\n")
print(total_sales_category)
print("\n")

# 3) Monthly sales trend
data['purchase_date'] = pd.to_datetime(data['purchase_date'])
data["Month"] = data["purchase_date"].dt.month_name()
monthly_sales = data.groupby("Month")["sales_amount"].sum()
print("--------------- Monthly sales trend ---------------")
print("\n")
print(monthly_sales)
print("\n")

# 4) Top 5 Most Profitable Products
top_profitable = data.nlargest(5, "profit")
print("--------------- Top 5 Most Profitable Products ---------------")
print("\n")
print(top_profitable[['order_id', 'product_category', 'profit', 'sales_amount']])
print("\n")

# 5) Customer age distribution
age_stats = data["customer_age"].describe()
print("--------------- Customer age distribution ---------------")
print("\n")
print(age_stats)
print("\n")

# 6) Best Performing Region by Sales
region_sales = data.groupby("region")["sales_amount"].sum()
print("--------------- Best Performing Region by Sales ---------------")
print("\n")
print(region_sales)
print("\n")

# 7) Correlation Between Sales, Profit, and Rating
correlation_matrix = data[['sales_amount', 'profit', 'rating']].corr()
print("--------------- Correlation Between Sales, Profit, and Rating ---------------")
print("\n")
print(correlation_matrix)
print("\n")

# 8) Total sales by product category Visualization

sns.set_style(style="whitegrid")
sns.set_palette("GnBu")
plt.figure(figsize=(10, 6))

ax = sns.barplot(data=data, x="product_category", y="sales_amount",
                 estimator="sum", errorbar=None)

plt.title("Total Sales by Product Category", fontweight='bold')
plt.xlabel("Product Category", fontsize=12)
plt.ylabel("Total Sales ($)", fontsize=12)

for container in ax.containers:
    ax.bar_label(container, fmt=lambda x: f'${x:,.0f}', padding=2)

plt.tight_layout()
plt.savefig("Visualization/sales_by_category_seaborn.png", dpi=300, bbox_inches='tight')
plt.show()
print("\n")

# 9) Monthly Sales Trend Visualization

sns.set_style(style="whitegrid")
sns.set_palette("viridis")
plt.figure(figsize=(10, 6))

sns.lineplot(data=data, x="Month", y="sales_amount",
            estimator="sum", marker="o")
plt.title("Monthly Sales Trend", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Months", fontsize=12)
plt.ylabel("Total Sales ($)", fontsize=12)

monthly_totals = data.groupby("Month")["sales_amount"].sum()
months = monthly_totals.index.tolist()
sales = monthly_totals.values.tolist()

for month, sale in zip(months, sales):
    plt.text(month, sale, f'${sale:,.0f}', 
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("Visualization/monthly_sales_trend_seaborn.png", dpi=300, bbox_inches='tight')
plt.show()
print("\n")

# 10) Top 5 Most Profitable Products Visualization

sns.set_style(style="whitegrid")
sns.set_palette("viridis")
plt.figure(figsize=(10, 6))

ax = sns.barplot(data=top_profitable, x="profit", y="order_id",
                hue="product_category", errorbar=None)

plt.title("Top 5 Most Profitable Products", fontweight='bold')
plt.xlabel("Profit", fontsize=12)
plt.ylabel("Order ID", fontsize=12)

for container in ax.containers:
    ax.bar_label(container, fmt=lambda x: f'${x:,.0f}', padding=2)

plt.legend(title='Product Category',
           bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.tight_layout()
plt.savefig("Visualization/top_5_profitable_product_seaborn.png", dpi=300, bbox_inches='tight')
plt.show()
print("\n")

# 11) Age Distribution Visualization

sns.set_style(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.histplot(data=data, x="customer_age",
            bins=20, kde=True, color="#4C72B0")

mean_age = data["customer_age"].mean()
plt.axvline(x=mean_age, color='blue', linestyle='--',
            label=f'Mean: {mean_age:.1f} years')

plt.title("Customer Age Distribution", fontweight='bold')
plt.xlabel("Customer Age (years)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.legend()

plt.tight_layout()
plt.savefig("Visualization/age_distribution_seaborn.png", dpi=300, bbox_inches='tight')
plt.show()
print("\n")

# 12) Best Performing Region by Sales Visualization

best_region = data.groupby("region")["sales_amount"].sum().idxmax()
best_value = data.groupby("region")["sales_amount"].sum().max()

sns.set_style(style="whitegrid")
sns.set_palette("Set3")
plt.figure(figsize=(10, 6))

ax = sns.barplot(data=data, x="region", y="sales_amount",
                estimator="sum", errorbar=None)

plt.title(f"Best Performing Region by Sales: {best_region} (${best_value:,.0f})",
          fontweight='bold')
plt.xlabel("Region", fontsize=12)
plt.ylabel("Total Sales ($)", fontsize=12)

for container in ax.containers:
    ax.bar_label(container, fmt=lambda x: f'${x:,.0f}', padding=2)

plt.tight_layout()
plt.savefig("Visualization/sales_by_region_seaborn.png", dpi=300, bbox_inches='tight')
plt.show()
print("\n")

# 13) Correlation Analysis Visualization

sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))

correlation_matrix = data[['sales_amount', 'profit', 'rating']].corr()
 
sns.heatmap(correlation_matrix,
            annot=True,
            linewidths=2,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8})

plt.title("Sales-Profit-Rating Correlation Analysis", fontweight='bold')
plt.xlabel("Business Metrics", fontsize=12)
plt.ylabel("Business Metrics", fontsize=12)

plt.tight_layout()
plt.savefig("Visualization/correlation_analysis_seaborn.png", dpi=300, bbox_inches='tight')
plt.show()