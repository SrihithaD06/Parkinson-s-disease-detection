import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessing import load_and_preprocess_data

X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
