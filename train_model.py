import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

FOLDERS = [
    "/home/nikodem/IVSPA/dislike_train",
    "/home/nikodem/IVSPA/fist_train",
    "/home/nikodem/IVSPA/like_train",
    "/home/nikodem/IVSPA/palm_train",
    "/home/nikodem/IVSPA/peace_train"
    # "/home/nikodem/IVSPA/dislike_test",
    # "/home/nikodem/IVSPA/fist_test",
    # "/home/nikodem/IVSPA/like_test",
    # "/home/nikodem/IVSPA/palm_test",
    # "/home/nikodem/IVSPA/peace_test"
]

FOLDSERS_VALID =[
    "/home/nikodem/IVSPA/dislike_valid",
    "/home/nikodem/IVSPA/fist_valid",
    "/home/nikodem/IVSPA/like_valid",
    "/home/nikodem/IVSPA/palm_valid",
    "/home/nikodem/IVSPA/peace_valid"
]


dataset = []
labels = []
dataset_valid = []
labels_valid = []


for dataPath in FOLDERS:
    
    label_name = dataPath.removesuffix('_train').split("/")[-1]
    #label_name = dataPath.removesuffix('_test').split("/")[-1]
    #print(labels)

    # read training file
    for file in os.listdir(dataPath):
        with open(f"{dataPath}/{file}", 'r') as f:
            data = f.read().removesuffix(";").split(';')
            data = np.array(data).astype(np.float32)
            dataset.append(data)
        labels.append(label_name)

#print(labels)


for dataPath in FOLDSERS_VALID:
    
    label_name = dataPath.removesuffix('_valid').split("/")[-1]
    #print(labels)

    # read training file
    for file in os.listdir(dataPath):
        with open(f"{dataPath}/{file}", 'r') as f:
            data = f.read().removesuffix(";").split(';')
            data = np.array(data).astype(np.float32)
            dataset_valid.append(data)
        labels_valid.append(label_name)

#print(labels)

classifier = RandomForestClassifier()

classifier.fit(dataset, labels)
score = classifier.score(dataset_valid, labels_valid)
print(f'Accuracy : {score}')
joblib.dump(classifier, f'model_finito.pkl')