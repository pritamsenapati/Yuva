import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Simulate dataset
np.random.seed(42)
n_samples = 1000

data = {
    'distance_km': np.random.uniform(5, 100, n_samples),
    'vehicle_type': np.random.choice(['Bike', 'Van', 'Truck'], n_samples),
    'traffic_level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'order_hour': np.random.randint(0, 24, n_samples),
    'day_of_week': np.random.randint(0, 7, n_samples),
    'weather_condition': np.random.choice(['Clear', 'Rain', 'Fog'], n_samples),
    'driver_experience_years': np.random.uniform(0, 10, n_samples),
    'warehouse_id': np.random.choice(['W1', 'W2', 'W3'], n_samples),
    'package_weight_kg': np.random.uniform(0.5, 20, n_samples)
}

df = pd.DataFrame(data)

# Simulate target variable with realistic relationships
df['delivery_time_hours'] = (
    0.5 * df['distance_km'] / 40 +  # Base speed 40 km/h
    np.where(df['traffic_level'] == 'High', 0.3, np.where(df['traffic_level'] == 'Medium', 0.15, 0)) +
    np.where(df['weather_condition'] == 'Rain', 0.2, np.where(df['weather_condition'] == 'Fog', 0.1, 0)) +
    np.where(df['vehicle_type'] == 'Bike', -0.1, np.where(df['vehicle_type'] == 'Truck', 0.2, 0)) +
    0.05 * df['package_weight_kg'] / 10 +
    np.random.normal(0, 0.3, n_samples)  # Noise
)

# Define features and target
X = df.drop('delivery_time_hours', axis=1)
y = df['delivery_time_hours']

# Preprocessing: encode categorical variables
categorical_features = ['vehicle_type', 'traffic_level', 'weather_condition', 'warehouse_id']
numerical_features = ['distance_km', 'order_hour', 'day_of_week', 'driver_experience_years', 'package_weight_kg']

preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
    print(f"{name}: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")