import joblib
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os

classifier: RandomForestClassifier = joblib.load('model_finito.pkl')
#classifier: RandomForestClassifier = joblib.load('model_test_proba.pkl')

FOLDERS = [
    "/home/nikodem/IVSPA/dislike_test",
    "/home/nikodem/IVSPA/fist_test",
    "/home/nikodem/IVSPA/like_test",
    "/home/nikodem/IVSPA/palm_test",
    "/home/nikodem/IVSPA/peace_test"
]

# #frame=cv2.imread('fist1/6.png')
# #frame=cv2.imread('dislike1/8.png')
# frame=cv2.imread('like.png')
# y_true = ['like']

dataset_all = []
labels_all = []

for dataPath in FOLDERS:
    dataset = []
    labels = []

    label_name = dataPath.removesuffix('_test').split("/")[-1]
    #print(labels)

    # read training file
    for file in os.listdir(dataPath):
        with open(f"{dataPath}/{file}", 'r') as f:
            data = f.read().removesuffix(";").split(';')
            data = np.array(data).astype(np.float32)
            dataset.append(data)
            dataset_all.append(data)
        labels.append(label_name)
        labels_all.append(label_name)
    if not dataset:
        continue
        
    result = classifier.predict(dataset)
    #print(result)

    accuracy = accuracy_score(labels, result, normalize=False)

    # print the accuracy score
    print(f"Accuracy for {label_name}: {accuracy/len(dataset)*100}%")
    print()

result = classifier.predict(dataset_all)
#print(result)

accuracy = accuracy_score(labels_all, result, normalize=False)

# print the accuracy score
print(f"Accuracy for model: {accuracy/len(dataset_all)*100}%")
print()

# Confusion_matrix
#tn, fp, fn, tp = confusion_matrix(labels_all, result, labels=list(set(labels_all))).ravel()
matrix = confusion_matrix(labels_all, result, labels=list(set(labels_all)))
print("Confusion matrix:")
print(matrix)
# print(f"True negative:       {tn}")
# print(f"False positive:      {fp}")
# print(f"False negative:      {fn}")
# print(f"True positive:       {tp}")
# print()
# print(f"Sensitivity:         {tp/(tp+fn)}")
# print(f"Specificity:         {tn/(fp+tn)}")
# print(f"Precision:           {tp/(tp+fp)}")
# print(f"Accuracy:            {(tp+tn)/(tn+fp+fn+tp)}")