import pickle
import numpy as np

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

sample = np.array([[119.992, 157.302, 74.997, 0.00784, 0.00007,
                    0.00370, 0.00554, 0.01109, 0.04374, 0.426,
                    0.02182, 0.03130, 0.02971, 0.06545, 0.02211,
                    21.033, 0.414783, 0.815285, -4.813031, 0.266482,
                    2.301442, 0.284654]])

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

if prediction[0] == 1:
    print("Parkinson's Detected")
else:
    print("Healthy")
