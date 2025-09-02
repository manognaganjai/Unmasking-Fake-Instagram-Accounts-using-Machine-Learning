import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv("final-v1.csv")

# Define features and target
features = [
    'username_length',
    'username_has_number',
    'full_name_has_number',
    'full_name_length',
    'is_private',
    'is_joined_recently',
    'has_channel',
    'is_business_account',
    'has_guides',
    'has_external_url',
    'edge_followed_by',
    'edge_follow'
]
target = 'is_fake'

X = df[features]
y = df[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Model Evaluation:\n")

print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and scaler
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/scam_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
