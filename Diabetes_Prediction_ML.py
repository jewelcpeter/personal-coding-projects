# done on Google Colab
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Step 2: Load the Dataset
diabetes = load_diabetes()
data = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
target = diabetes.target
# Display the dataset description
print(diabetes.DESCR)
# Step 3: Explore the Data
print("\nDataset Shape:", data.shape)
print("\nFeature Names:", diabetes.feature_names)
print("\nDataset Preview:\n", data.head())
print("\nTarget Preview (Progression of diabetes disease):\n", target[:5])
# Step 4: Data Preprocessing
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
X_train, X_test, y_train, y_test = train_test_split(data_scaled, target, test_size=0.2, random_state=42)
# Step 5: Build the Model
model = LinearRegression()
# Step 6: Train the Model
model.fit(X_train, y_train)
# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMean Squared Error:", mse)
print("R-squared Score:", r2)
# Step 8: Make Predictions
sample_data = X_test[0].reshape(1, -1)  # Take a sample from the test set
sample_prediction = model.predict(sample_data)
print("\nSample Prediction:", sample_prediction)
