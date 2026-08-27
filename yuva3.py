import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

# ============================================================================
# 2. DATA LOADING & INITIAL INSPECTION
# ============================================================================

# Read the CSV file
df = pd.read_csv('Delivery_Logistics.csv')

print("=" * 80)
print("1. INITIAL DATA INSPECTION")
print("=" * 80)

# Basic information
print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn Names:\n{df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 Rows:\n{df.head()}")

# Check for missing values
print(f"\nMissing Values:\n{df.isnull().sum()}")

# Summary statistics for numerical columns
print(f"\nNumerical Columns Summary:\n{df.describe()}")

# ============================================================================
# 3. DATA CLEANING & PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("2. DATA CLEANING & PREPROCESSING")
print("=" * 80)

# 3.1 Drop constant column (delivery_id is constant 250.99 or similar)
if 'delivery_id' in df.columns:
    print(f"\nDropping 'delivery_id' column (constant value: {df['delivery_id'].unique()[:3]})")
    df = df.drop('delivery_id', axis=1)

# 3.2 Convert date columns to datetime and extract meaningful time
print("\nConverting date-time columns...")
df['delivery_time_hours'] = pd.to_datetime(df['delivery_time_hours'], errors='coerce')
df['expected_time_hours'] = pd.to_datetime(df['expected_time_hours'], errors='coerce')

# 3.3 Create delivery speed (difference between actual and expected)
df['delivery_speed_seconds'] = (df['delivery_time_hours'] - df['expected_time_hours']).dt.total_seconds()

# 3.4 Create a categorical 'delivery_speed' column
df['delivery_speed_category'] = np.where(
    df['delivery_speed_seconds'] <= 0,
    'On-Time',
    'Delayed'
)

print(f"\nDelivery Speed Summary:")
print(df['delivery_speed_category'].value_counts())

# 3.5 Extract time components if needed
df['delivery_hour'] = df['delivery_time_hours'].dt.hour
df['delivery_day'] = df['delivery_time_hours'].dt.dayofweek

# 3.6 Clean up 'delayed' column - it might have inconsistencies
print(f"\nUnique values in 'delayed' column before cleaning: {df['delayed'].unique()}")
df['delayed'] = df['delayed'].map({'yes': 1, 'no': 0, 'y': 1, 'n': 0}).fillna(0).astype(int)

# 3.7 Clean 'delivery_status' column
print(f"\nUnique values in 'delivery_status': {df['delivery_status'].unique()}")

# 3.8 Convert categorical columns to category type
categorical_cols = ['delivery_partner', 'package_type', 'vehicle_type', 
                    'delivery_mode', 'region', 'weather_condition', 
                    'delivery_status', 'delivery_speed_category']

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# 3.9 Create a failure flag
df['failed'] = (df['delivery_status'] == 'failed').astype(int)

print(f"\nFinal Dataset Shape: {df.shape}")

# ============================================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# ============================================================================
# 4.1 UNIVARIATE ANALYSIS (Single Variables)
# ============================================================================

print("\n" + "-" * 60)
print("4.1 UNIVARIATE ANALYSIS")
print("-" * 60)

# 4.1.1 Distribution of Categorical Variables
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Distribution of Key Categorical Variables', fontsize=16)

# Delivery Partners
df['delivery_partner'].value_counts().plot(kind='bar', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Delivery Partners')
axes[0,0].set_xlabel('Partner')
axes[0,0].set_ylabel('Count')
axes[0,0].tick_params(axis='x', rotation=45)

# Package Types
df['package_type'].value_counts().head(10).plot(kind='bar', ax=axes[0,1], color='lightgreen')
axes[0,1].set_title('Package Types (Top 10)')
axes[0,1].set_xlabel('Package Type')
axes[0,1].set_ylabel('Count')
axes[0,1].tick_params(axis='x', rotation=45)

# Vehicle Types
df['vehicle_type'].value_counts().plot(kind='bar', ax=axes[0,2], color='salmon')
axes[0,2].set_title('Vehicle Types')
axes[0,2].set_xlabel('Vehicle Type')
axes[0,2].set_ylabel('Count')
axes[0,2].tick_params(axis='x', rotation=45)

# Delivery Modes
df['delivery_mode'].value_counts().plot(kind='bar', ax=axes[1,0], color='orchid')
axes[1,0].set_title('Delivery Modes')
axes[1,0].set_xlabel('Delivery Mode')
axes[1,0].set_ylabel('Count')
axes[1,0].tick_params(axis='x', rotation=45)

# Regions
df['region'].value_counts().plot(kind='bar', ax=axes[1,1], color='gold')
axes[1,1].set_title('Regions')
axes[1,1].set_xlabel('Region')
axes[1,1].set_ylabel('Count')
axes[1,1].tick_params(axis='x', rotation=45)

# Weather Conditions
df['weather_condition'].value_counts().plot(kind='bar', ax=axes[1,2], color='lightcoral')
axes[1,2].set_title('Weather Conditions')
axes[1,2].set_xlabel('Weather Condition')
axes[1,2].set_ylabel('Count')
axes[1,2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('EDA_1_Categorical_Distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.1.2 Distribution of Numerical Variables
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribution of Key Numerical Variables', fontsize=16)

# Delivery Cost
df['delivery_cost'].hist(bins=50, ax=axes[0,0], color='steelblue', edgecolor='black')
axes[0,0].set_title('Delivery Cost Distribution')
axes[0,0].set_xlabel('Cost ($)')
axes[0,0].set_ylabel('Frequency')

# Distance
df['distance_km'].hist(bins=50, ax=axes[0,1], color='forestgreen', edgecolor='black')
axes[0,1].set_title('Distance Distribution')
axes[0,1].set_xlabel('Distance (km)')
axes[0,1].set_ylabel('Frequency')

# Package Weight
df['package_weight_kg'].hist(bins=50, ax=axes[0,2], color='darkorange', edgecolor='black')
axes[0,2].set_title('Package Weight Distribution')
axes[0,2].set_xlabel('Weight (kg)')
axes[0,2].set_ylabel('Frequency')

# Delivery Speed
df['delivery_speed_seconds'].hist(bins=50, ax=axes[1,0], color='crimson', edgecolor='black')
axes[1,0].set_title('Delivery Speed Distribution')
axes[1,0].set_xlabel('Speed (seconds)')
axes[1,0].set_ylabel('Frequency')

# Delivery Rating
df['delivery_rating'].value_counts().sort_index().plot(kind='bar', ax=axes[1,1], color='mediumpurple')
axes[1,1].set_title('Delivery Rating Distribution')
axes[1,1].set_xlabel('Rating (1-5)')
axes[1,1].set_ylabel('Count')

# Delivery Status
df['delivery_status'].value_counts().plot(kind='pie', ax=axes[1,2], autopct='%1.1f%%', explode=[0.02]*len(df['delivery_status'].unique()))
axes[1,2].set_title('Delivery Status Distribution')
axes[1,2].set_ylabel('')

plt.tight_layout()
plt.savefig('EDA_2_Numerical_Distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.1.3 Summary Statistics
print("\nSummary Statistics for Numerical Variables:")
print(df[['distance_km', 'package_weight_kg', 'delivery_cost', 'delivery_speed_seconds', 'delivery_rating']].describe())

# ============================================================================
# 4.2 BIVARIATE ANALYSIS (Relationships)
# ============================================================================

print("\n" + "-" * 60)
print("4.2 BIVARIATE ANALYSIS")
print("-" * 60)

# 4.2.1 Correlation Matrix
fig, ax = plt.subplots(figsize=(12, 8))
numerical_cols = ['distance_km', 'package_weight_kg', 'delivery_cost', 'delivery_rating', 
                 'delivery_speed_seconds', 'delayed', 'failed']
corr_matrix = df[numerical_cols].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix - Numerical Variables', fontsize=16)
plt.tight_layout()
plt.savefig('EDA_3_Correlation_Matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.2.2 Scatter Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Cost vs Distance
scatter1 = axes[0].scatter(df['distance_km'], df['delivery_cost'], alpha=0.5, c=df['delivery_partner'].cat.codes, cmap='viridis')
axes[0].set_title('Delivery Cost vs Distance')
axes[0].set_xlabel('Distance (km)')
axes[0].set_ylabel('Cost ($)')
plt.colorbar(scatter1, ax=axes[0], label='Partner')

# Cost vs Weight
scatter2 = axes[1].scatter(df['package_weight_kg'], df['delivery_cost'], alpha=0.5, c=df['delivery_mode'].cat.codes, cmap='plasma')
axes[1].set_title('Delivery Cost vs Package Weight')
axes[1].set_xlabel('Weight (kg)')
axes[1].set_ylabel('Cost ($)')
plt.colorbar(scatter2, ax=axes[1], label='Delivery Mode')

# Cost vs Distance colored by Delay
scatter3 = axes[2].scatter(df['distance_km'], df['delivery_cost'], alpha=0.5, c=df['delayed'], cmap='bwr')
axes[2].set_title('Delivery Cost vs Distance (Colored by Delay)')
axes[2].set_xlabel('Distance (km)')
axes[2].set_ylabel('Cost ($)')
plt.colorbar(scatter3, ax=axes[2], label='Delayed (1=Yes, 0=No)')

plt.tight_layout()
plt.savefig('EDA_4_Scatter_Plots.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.2.3 Box Plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Cost by Partner
df.boxplot(column='delivery_cost', by='delivery_partner', ax=axes[0,0])
axes[0,0].set_title('Cost by Delivery Partner')
axes[0,0].set_xlabel('Partner')
axes[0,0].set_ylabel('Cost ($)')

# Cost by Vehicle Type
df.boxplot(column='delivery_cost', by='vehicle_type', ax=axes[0,1])
axes[0,1].set_title('Cost by Vehicle Type')
axes[0,1].set_xlabel('Vehicle Type')
axes[0,1].set_ylabel('Cost ($)')

# Cost by Delivery Mode
df.boxplot(column='delivery_cost', by='delivery_mode', ax=axes[0,2])
axes[0,2].set_title('Cost by Delivery Mode')
axes[0,2].set_xlabel('Delivery Mode')
axes[0,2].set_ylabel('Cost ($)')

# Cost by Region
df.boxplot(column='delivery_cost', by='region', ax=axes[1,0])
axes[1,0].set_title('Cost by Region')
axes[1,0].set_xlabel('Region')
axes[1,0].set_ylabel('Cost ($)')

# Cost by Weather
df.boxplot(column='delivery_cost', by='weather_condition', ax=axes[1,1])
axes[1,1].set_title('Cost by Weather Condition')
axes[1,1].set_xlabel('Weather Condition')
axes[1,1].set_ylabel('Cost ($)')

# Cost by Package Type (Top 5)
top_packages = df['package_type'].value_counts().head(5).index
df_subset = df[df['package_type'].isin(top_packages)]
df_subset.boxplot(column='delivery_cost', by='package_type', ax=axes[1,2])
axes[1,2].set_title('Cost by Package Type (Top 5)')
axes[1,2].set_xlabel('Package Type')
axes[1,2].set_ylabel('Cost ($)')

plt.suptitle('Delivery Cost by Categorical Variables')
plt.tight_layout()
plt.savefig('EDA_5_Box_Plots.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.2.4 Performance Metrics by Category
print("\n" + "-" * 60)
print("4.2.4 Performance Metrics by Category")
print("-" * 60)

# Performance by Partner
print("\nDelivery Performance by Partner:")
partner_performance = df.groupby('delivery_partner').agg({
    'delivery_cost': 'mean',
    'delayed': 'mean',
    'failed': 'mean',
    'distance_km': 'mean',
    'package_weight_kg': 'mean'
}).round(2)
partner_performance.columns = ['Avg_Cost', 'Delay_Rate', 'Failure_Rate', 'Avg_Distance', 'Avg_Weight']
print(partner_performance.sort_values('Delay_Rate', ascending=True))

# Performance by Delivery Mode
print("\n\nDelivery Performance by Delivery Mode:")
mode_performance = df.groupby('delivery_mode').agg({
    'delivery_cost': 'mean',
    'delayed': 'mean',
    'failed': 'mean',
    'distance_km': 'mean',
    'package_weight_kg': 'mean'
}).round(2)
mode_performance.columns = ['Avg_Cost', 'Delay_Rate', 'Failure_Rate', 'Avg_Distance', 'Avg_Weight']
print(mode_performance.sort_values('Delay_Rate', ascending=True))

# Performance by Weather
print("\n\nDelivery Performance by Weather Condition:")
weather_performance = df.groupby('weather_condition').agg({
    'delivery_cost': 'mean',
    'delayed': 'mean',
    'failed': 'mean'
}).round(2)
weather_performance.columns = ['Avg_Cost', 'Delay_Rate', 'Failure_Rate']
print(weather_performance.sort_values('Delay_Rate', ascending=True))

# Performance by Region
print("\n\nDelivery Performance by Region:")
region_performance = df.groupby('region').agg({
    'delivery_cost': 'mean',
    'delayed': 'mean',
    'failed': 'mean'
}).round(2)
region_performance.columns = ['Avg_Cost', 'Delay_Rate', 'Failure_Rate']
print(region_performance.sort_values('Delay_Rate', ascending=True))

# ============================================================================
# 4.3 MULTIVARIATE ANALYSIS
# ============================================================================

print("\n" + "-" * 60)
print("4.3 MULTIVARIATE ANALYSIS")
print("-" * 60)

# 4.3.1 Heatmap - Delay Rate by Vehicle & Package Type
print("\nCreating Delay Rate Heatmap by Vehicle & Package Type...")
delay_pivot = df.groupby(['vehicle_type', 'package_type'])['delayed'].mean().unstack()

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(delay_pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0.3, ax=ax,
            cbar_kws={'label': 'Delay Rate'})
ax.set_title('Delay Rate by Vehicle Type and Package Type', fontsize=16)
ax.set_xlabel('Package Type')
ax.set_ylabel('Vehicle Type')
plt.tight_layout()
plt.savefig('EDA_6_Heatmap_Vehicle_Package.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.3.2 Box Plot - Cost by Partner and Mode
fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(x='delivery_partner', y='delivery_cost', hue='delivery_mode', data=df, ax=ax)
ax.set_title('Delivery Cost by Partner and Mode', fontsize=14)
ax.set_xlabel('Delivery Partner')
ax.set_ylabel('Cost ($)')
ax.legend(title='Delivery Mode', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('EDA_7_Cost_Partner_Mode.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.3.3 Delay Rate by Partner & Weather
print("\nCreating Partner & Weather Performance Chart...")
partner_weather = df.groupby(['delivery_partner', 'weather_condition']).agg({
    'delayed': 'mean',
    'delivery_cost': 'mean'
}).round(2).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Delay Rate by Partner & Weather
pivot_delay = partner_weather.pivot(index='delivery_partner', columns='weather_condition', values='delayed')
pivot_delay.plot(kind='bar', ax=axes[0], colormap='RdYlGn_r', ylim=(0, 0.5))
axes[0].set_title('Delay Rate by Partner and Weather Condition', fontsize=14)
axes[0].set_xlabel('Delivery Partner')
axes[0].set_ylabel('Delay Rate')
axes[0].legend(title='Weather', bbox_to_anchor=(1.05, 1))
axes[0].tick_params(axis='x', rotation=45)

# Average Cost by Partner & Weather
pivot_cost = partner_weather.pivot(index='delivery_partner', columns='weather_condition', values='delivery_cost')
pivot_cost.plot(kind='bar', ax=axes[1], colormap='viridis')
axes[1].set_title('Average Cost by Partner and Weather Condition', fontsize=14)
axes[1].set_xlabel('Delivery Partner')
axes[1].set_ylabel('Average Cost ($)')
axes[1].legend(title='Weather', bbox_to_anchor=(1.05, 1))
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('EDA_8_Partner_Weather.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.3.4 Delivery Speed by Category
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Speed by Partner
df.boxplot(column='delivery_speed_seconds', by='delivery_partner', ax=axes[0,0])
axes[0,0].set_title('Delivery Speed by Partner')
axes[0,0].set_xlabel('')
axes[0,0].set_ylabel('Speed (seconds)')

# Speed by Mode
df.boxplot(column='delivery_speed_seconds', by='delivery_mode', ax=axes[0,1])
axes[0,1].set_title('Delivery Speed by Mode')
axes[0,1].set_xlabel('')
axes[0,1].set_ylabel('Speed (seconds)')

# Speed by Weather
df.boxplot(column='delivery_speed_seconds', by='weather_condition', ax=axes[1,0])
axes[1,0].set_title('Delivery Speed by Weather')
axes[1,0].set_xlabel('')
axes[1,0].set_ylabel('Speed (seconds)')

# Speed by Region
df.boxplot(column='delivery_speed_seconds', by='region', ax=axes[1,1])
axes[1,1].set_title('Delivery Speed by Region')
axes[1,1].set_xlabel('')
axes[1,1].set_ylabel('Speed (seconds)')

plt.suptitle('Delivery Speed (seconds) by Categorical Variables')
plt.tight_layout()
plt.savefig('EDA_9_Speed_Analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 5. ADVANCED ANALYTICAL INSIGHTS
# ============================================================================

print("\n" + "=" * 80)
print("4. ADVANCED ANALYTICAL INSIGHTS")
print("=" * 80)

# 5.1 Top 10 Most Costly Deliveries
print("\nTop 10 Most Expensive Deliveries:")
print(df.nlargest(10, 'delivery_cost')[['delivery_partner', 'package_type', 'vehicle_type', 
                                       'region', 'distance_km', 'package_weight_kg', 
                                       'delivery_cost']])

# 5.2 Bottom 10 Least Costly Deliveries
print("\nBottom 10 Least Expensive Deliveries:")
print(df.nsmallest(10, 'delivery_cost')[['delivery_partner', 'package_type', 'vehicle_type', 
                                        'region', 'distance_km', 'package_weight_kg', 
                                        'delivery_cost']])

# 5.3 Average Cost per kg per km
df['cost_per_kg'] = df['delivery_cost'] / df['package_weight_kg']
df['cost_per_km'] = df['delivery_cost'] / df['distance_km']

print("\nAverage Cost Metrics:")
print(f"Average Cost per kg: ${df['cost_per_kg'].mean():.2f}")
print(f"Average Cost per km: ${df['cost_per_km'].mean():.2f}")

print("\nCost Efficiency by Partner (Cost per km):")
print(df.groupby('delivery_partner')['cost_per_km'].mean().sort_values())

# 5.4 High Risk Deliveries (Heavy + Long Distance)
df['heavy_long'] = ((df['package_weight_kg'] > df['package_weight_kg'].median()) & 
                    (df['distance_km'] > df['distance_km'].median()))

print(f"\nHigh Risk Deliveries (Heavy & Long Distance): {df['heavy_long'].sum()} out of {len(df)} ({df['heavy_long'].mean()*100:.1f}%)")
print(f"Delay Rate for High Risk Deliveries: {df[df['heavy_long']]['delayed'].mean()*100:.1f}%")
print(f"Delay Rate for Low Risk Deliveries: {df[~df['heavy_long']]['delayed'].mean()*100:.1f}%")

# 5.5 Rating Analysis
print("\nRating Analysis:")
rating_df = df.groupby('delivery_speed_category')['delivery_rating'].agg(['mean', 'count'])
rating_df['percentage'] = (rating_df['count'] / len(df)) * 100
print(rating_df)

# 5.6 Partner vs Mode Performance Matrix
print("\nPartner vs Mode Performance Matrix (Delay Rate):")
partner_mode_delay = df.groupby(['delivery_partner', 'delivery_mode'])['delayed'].mean().unstack().round(2)
print(partner_mode_delay)

# ============================================================================
# 6. SUMMARY INSIGHTS & RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("5. SUMMARY INSIGHTS & RECOMMENDATIONS")
print("=" * 80)
