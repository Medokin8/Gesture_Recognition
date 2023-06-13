import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

MODELS_TRIANED = [
    'model_10.pkl',
    'model_50.pkl',
    'model_100.pkl',
    'model_400.pkl',
    'model_630.pkl'
]


FOLDERS = [
    "/home/nikodem/IVSPA/dislike_test",
    "/home/nikodem/IVSPA/fist_test",
    "/home/nikodem/IVSPA/like_test",
    "/home/nikodem/IVSPA/palm_test",
    "/home/nikodem/IVSPA/peace_test"
]

dataset_all = []
labels_all = []

for dataPath in FOLDERS:

    label_name = dataPath.removesuffix('_test').split("/")[-1]
    #print(labels)

    # read training file
    for file in os.listdir(dataPath):
        with open(f"{dataPath}/{file}", 'r') as f:
            data = f.read().removesuffix(";").split(';')
            data = np.array(data).astype(np.float32)
            dataset_all.append(data)
        labels_all.append(label_name)

for num_model in MODELS_TRIANED :
    classifier: RandomForestClassifier = joblib.load(num_model)
    #classifier: RandomForestClassifier = joblib.load('model_test_proba.pkl')  
      
    print(f'Classifier {num_model}')
    # Confusion Matrix
    predictions = classifier.predict(dataset_all)
    cm = confusion_matrix(labels_all, predictions)
    print("Confusion Matrix:")
    print(cm)
    print()

    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(set(labels_all)))
    #disp.plot()
    #plt.show()

    # Classification Report
    cr = classification_report(labels_all, predictions, zero_division = 1)
    print("Classification Report:")
    print(cr)
    print()