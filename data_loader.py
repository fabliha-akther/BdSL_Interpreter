import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_dataset(folder_path):
    allowed_labels = {"Sign 0", "Sign 1", "Sign 2", "Sign 3", "Sign 4",
                      "sign00", "sign01", "sign02", "sign03", "sign04"}
    X = []
    y = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print("Found file:", file_path)

                try:
                    data = pd.read_csv(file_path)

                    if data.shape[1] < 64:
                        continue  

                    
                    flattened = data.iloc[0, 1:64].values.astype(np.float32)

                    label = os.path.basename(os.path.dirname(file_path))
                    if flattened.shape[0] == 63 and label in allowed_labels:
                         X.append(flattened)
                         y.append(label)


                except Exception as e:
                    print("Skipped:", file_path, "|", e)
                    continue

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return np.array(X), np.array(y_encoded), le
