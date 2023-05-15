import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

FOLDERS = [
    "/home/nikodem/IVSPA/dislike_ready",
    "/home/nikodem/IVSPA/fist_ready",
    "/home/nikodem/IVSPA/like_ready",
    "/home/nikodem/IVSPA/palm_ready",
    "/home/nikodem/IVSPA/peace_ready"
]

dataset = []
labels = []

for dataPath in FOLDERS:
    
    label_name = dataPath.removesuffix('_ready').split("/")[-1]
    #print(labels)

    # read training file
    for file in os.listdir(dataPath):
        with open(f"{dataPath}/{file}", 'r') as f:
            data = f.read().removesuffix(";").split(';')
            data = np.array(data).astype(np.float32)
            dataset.append(data)
        labels.append(label_name)

#print(labels)

classifier = RandomForestClassifier()

classifier.fit(dataset, labels)
joblib.dump(classifier, f'model.pkl')
