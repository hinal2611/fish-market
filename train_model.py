import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the Dataset
df = pd.read_csv("Fish.csv")
# Preprocessing
X = df.drop(['Species'], axis=1)
y = df['Species']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the Model
with open('fish_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as fish_model.pkl")