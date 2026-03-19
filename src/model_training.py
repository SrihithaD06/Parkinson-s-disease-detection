import pickle
from sklearn.svm import SVC
from data_preprocessing import load_and_preprocess_data

X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("../data/parkinsons.csv")

model = SVC(kernel='rbf', C=10, gamma='scale')
model.fit(X_train, y_train)

with open("../models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")
