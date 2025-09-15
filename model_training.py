# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

# --- 1. Load the Dataset ---
# Ensure your dataset.csv is in the same directory as this script
df = pd.read_csv('dataset.csv')

# --- 2. Clean and Encode the Target Variable ---
df['Target'] = df['Target'].map({'Dropout': 1, 'Graduate': 0, 'Enrolled': 0})

# --- 3. Select Features for the Model ---
# This is the key change! We're now only selecting a few features.
selected_features = [
    'Age at enrollment',
    'Marital status',
    'Scholarship holder',
    'Tuition fees up to date',
    'Gender',
    'Previous qualification',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    "Father's qualification",
    "Mother's qualification",
    'Daytime/evening attendance',
    'Unemployment rate'
]

X = df[selected_features]
y = df['Target']

# --- 4. Identify Feature Types and Create Preprocessing Pipelines ---
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- 5. Define and Train the Models ---
models = {
    'LogisticRegression': LogisticRegression(random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42)
}

best_model = None
best_score = 0

for name, model in models.items():
    print(f"Training {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    mean_accuracy = scores.mean()
    print(f"{name} 5-fold Cross-Validation Accuracy: {mean_accuracy:.4f}\n")
    if mean_accuracy > best_score:
        best_score = mean_accuracy
        best_model = pipeline
        best_model_name = name

# --- 6. Final Training and Saving the Best Model ---
print(f"Training the best model ({best_model_name}) on the full dataset...")
best_model.fit(X, y)

with open('dropout_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("Best model saved as 'dropout_model.pkl'.")