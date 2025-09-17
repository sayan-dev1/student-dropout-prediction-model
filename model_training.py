# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

# --- 1. Load the Dataset ---
# Ensure your dataset.csv is in the same directory as this script
df = pd.read_csv('dataset.csv')

# --- 2. Clean and Encode the Target Variable ---
df['Target'] = df['Target'].map({'Dropout': 1, 'Graduate': 0, 'Enrolled': 0})

# --- 3. Select Features for the Model ---
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

# --- NEW: Split the data for training and testing ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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
best_model_name = ""

for name, model in models.items():
    print(f"Training {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    # We use cross_val_score on the training data to find the best model
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    mean_accuracy = scores.mean()
    print(f"{name} 5-fold Cross-Validation Accuracy: {mean_accuracy:.4f}\n")
    if mean_accuracy > best_score:
        best_score = mean_accuracy
        best_model = pipeline
        best_model_name = name

# --- 6. Final Training and Saving the Best Model ---
print(f"Training the best model ({best_model_name}) on the full training dataset...")
best_model.fit(X_train, y_train)

# --- NEW: Calculate metrics on the test set and save to a file ---
print("Calculating and saving model evaluation metrics...")
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall
}

with open('model_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)
print("Model metrics saved as 'model_metrics.pkl'.")

# --- NEW: Extract and save feature importance ---
print("Extracting and saving feature importance...")
try:
    # Access the classifier from the pipeline to get feature importances
    classifier = best_model.named_steps['classifier']

    # Get the feature names after one-hot encoding
    feature_names_out = best_model.named_steps['preprocessor'].get_feature_names_out()
    
    # Create a DataFrame for the importance scores
    importance_df = pd.DataFrame({
        'Feature': feature_names_out,
        'Importance': classifier.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Save the DataFrame to a pickle file
    with open('feature_importance.pkl', 'wb') as f:
        pickle.dump(importance_df, f)
    print("Feature importance saved as 'feature_importance.pkl'.")

except AttributeError:
    # Handle cases where the best model does not have feature_importances_
    print(f"The selected model ({best_model_name}) does not support feature importance.")
    with open('feature_importance.pkl', 'wb') as f:
        pickle.dump(None, f)

# --- 7. Save the best model to a file ---
with open('dropout_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
print("Best model saved as 'dropout_model.pkl'.")
