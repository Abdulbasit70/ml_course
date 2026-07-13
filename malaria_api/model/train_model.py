# model/train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Simulate malaria dataset
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'age':         np.random.randint(1, 80, n),
    'fever':       np.random.randint(0, 1, n),        # 0=no 1=yes
    'chills':      np.random.randint(0, 1, n),
    'headache':    np.random.randint(0, 1, n),
    'temperature': np.random.uniform(36.0, 41.0, n),  # body temp °C
    'malaria':     np.random.randint(0, 2, n)          # target
})

X = data.drop('malaria', axis=1)
y = data['malaria']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    class_weight='balanced',   # handles imbalance
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print(classification_report(y_test, preds))

# Save model to disk using pickle
with open('malaria_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved successfully!")