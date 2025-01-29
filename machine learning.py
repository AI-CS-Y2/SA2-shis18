import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('output.csv')

# Drop rows with missing target values
target_column = 'price'
data = data.dropna(subset=[target_column])

# Fill missing values in other columns using forward and backward fill
data = data.ffill().bfill()

# Encode categorical columns
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split dataset into features and target
X = data.drop(columns=[target_column])
y = data[target_column]

# Standardize features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate models
lin_reg_mse = mean_squared_error(y_test, lin_reg.predict(X_test))
knn_mse = mean_squared_error(y_test, knn.predict(X_test))

print("Linear Regression MSE:", lin_reg_mse)
print("KNN MSE:", knn_mse)

# Predict on new data
# Define new data with valid feature names
new_data = pd.DataFrame(
    [[0, 1, 0, 4.0, 1, 2184.0, 11003.0, 1.0, 0, 0, 15.0, 100.0, 5.0]],  # Replace with actual new data
    columns=X.columns  # Use feature names matching training data
)

# Scale the new data
new_data_scaled = pd.DataFrame(scaler.transform(new_data), columns=X.columns)

# Make predictions
lin_reg_prediction = lin_reg.predict(new_data_scaled)
knn_prediction = knn.predict(new_data_scaled)

print("Linear Regression Prediction:", lin_reg_prediction)
print("KNN Prediction:", knn_prediction)

