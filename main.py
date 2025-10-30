# import files
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# 1. LOAD DATASET
file_path = "DATASET/channel_quality_dataset.csv"
df = pd.read_csv(file_path)
print(" Dataset loaded successfully!\n")
print(df.head())

# 2. HANDLE MISSING VALUES & CLEANING
df = df.dropna()
df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Encode target column
le = LabelEncoder()
df["Channel_Quality"] = le.fit_transform(df["Channel_Quality"].astype(str))

# 3. FEATURE AND TARGET SELECTION
X = df[["RSSI", "SNR", "BER", "Distance", "Interference"]]  # removed Channel_Quality from X
y = df["Channel_Quality"]

# 4. SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. TRAIN MODEL
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. PREDICT TEST DATA
y_pred = model.predict(X_test)

# 7. EVALUATE
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Model trained successfully!")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# 8. FINAL MODEL PREDICTION
sample = [[-70, 25, 0.01, 100, 0.2]]
pred = model.predict(sample)
pred_label = le.inverse_transform([int(round(pred[0]))])
print("Predicted Channel Quality:", pred_label[0])
