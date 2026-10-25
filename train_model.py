
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load dataset
df = pd.read_csv("House_Rent_Dataset.csv")

# 2. Select input features and target
features = [
    "BHK",
    "Size",
    "City",
    "Furnishing Status",
    "Bathroom",
    "Area Type"
]
target = "Rent"

# 3. Keep required columns and clean data
df = df[features + [target]].copy()

for col in ["BHK", "Size", "Bathroom", "Rent"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[target])
df = df[df[target] > 0]
df = df.drop_duplicates()

X = df[features]
y = df[target]

# 4. Identify numerical and categorical features
numeric_features = ["BHK", "Size", "Bathroom"]
categorical_features = [
    "City",
    "Furnishing Status",
    "Area Type"
]

# 5. Preprocess the input data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 6. Create the Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    ))
])

# 7. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train the model
model.fit(X_train, y_train)

# 9. Evaluate the model
predictions = model.predict(X_test)

rmse = mean_squared_error(y_test, predictions) ** 0.5
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.3f}")

# 10. Save the complete trained pipeline
with open("rent_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")
