import lightgbm as lgb
import numpy as np
import json
import os

# 1. Test Pathing
model_path = "app/models/sensor_lgbm_model.txt"
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
else:
    print(f"SUCCESS: Model found at {model_path}")

# 2. Load and Force Predict
try:
    bst = lgb.Booster(model_file=model_path)
    
    # High Stress Vector: [Temp, Vib, Pres, Flow, RPM, Hours]
    # We use a 2D array because LightGBM expects a batch
    test_input = np.array([[120.0, 15.0, 0.0, 5.0, 4500.0, 5000.0]])
    
    prediction = bst.predict(test_input)
    print(f"DIRECT PREDICTION RESULT: {prediction[0]}")
    
    if prediction[0] == 0:
        print("The model itself is returning 0. Check the training target (Maintenance_Flag).")
except Exception as e:
    print(f"CRITICAL ERROR DURING PREDICTION: {e}")