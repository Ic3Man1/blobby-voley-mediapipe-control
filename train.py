import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('hand_gestures.csv', header=None)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Dokładność modelu: {accuracy_score(y_test, y_pred) * 100:.2f}%")

with open('hand_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model zapisany jako hand_model.pkl")