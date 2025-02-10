import pickle
import numpy as np
import sys

if len(sys.argv) < 25:  # script name + 24 values
    print("Please provide 24 values separated by spaces")
    print("Example: python test_depression.py 31 2 26265.67 1 0 0 1 0 0 0 0 1 0 0 1 0 1 1 0 0 0 0 1 1")
    sys.exit(1)

try:
    
    with open('lightgbm_model.pkl', 'rb') as f:
        model = pickle.load(f)

    values = [float(x) for x in sys.argv[1:]]
    X_new = np.array([values])

    prob = model.predict(X_new)[0]
    prediction = 1 if prob >= 0.5 else 0

    print(f"Probability: {prob:.4f}")
    print(f"Prediction: {prediction}")

except Exception as e:
    print(f"Error: {str(e)}")