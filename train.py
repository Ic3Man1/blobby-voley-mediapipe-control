import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('hand_gestures.csv', header=None)

y = df.iloc[:, 0]
X = df.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
print(f"accuracy score {accuracy_score(y_test, y_predict)}")

with open('hand_model.pkl', 'wb') as file:
    pickle.dump(clf, file)