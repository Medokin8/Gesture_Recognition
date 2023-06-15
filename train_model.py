import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('TKAgg')

FOLDERS = [
    "/home/nikodem/IVSPA/dislike_train",
    "/home/nikodem/IVSPA/fist_train",
    "/home/nikodem/IVSPA/like_train",
    "/home/nikodem/IVSPA/palm_train",
    "/home/nikodem/IVSPA/peace_train"
]

FOLDSERS_VALID =[
    "/home/nikodem/IVSPA/dislike_valid",
    "/home/nikodem/IVSPA/fist_valid",
    "/home/nikodem/IVSPA/like_valid",
    "/home/nikodem/IVSPA/palm_valid",
    "/home/nikodem/IVSPA/peace_valid"
]

INT_FILES = [10, 50, 100, 200, 400, 500, 630]

dataset_valid = []
labels_valid = []
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


for number in INT_FILES:
    Num_files = number #630 maks
    num_folds = 5  # Number of cross-validation folds

    dataset = []
    labels = []


    for dataPath in FOLDERS:
        
        label_name = dataPath.removesuffix('_train').split("/")[-1]
        #label_name = dataPath.removesuffix('_test').split("/")[-1]
        #print(labels)
        
        i=0
        
        # read training file
        for file in os.listdir(dataPath):
            if i >= Num_files :
                break
            with open(f"{dataPath}/{file}", 'r') as f:
                data = f.read().removesuffix(";").split(';')
                data = np.array(data).astype(np.float32)
                dataset.append(data)
            labels.append(label_name)
            i = i+1

    #print(labels)

    classifier = RandomForestClassifier()

    # Cross-validation training
    scores = cross_val_score(classifier, dataset, labels, cv=num_folds)
    print(f'Cross-Validation Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})')

    classifier.fit(dataset, labels)

    # score = classifier.score(dataset_valid, labels_valid)
    # print(f'Accuracy : {score}')

    joblib.dump(classifier, f'model_{Num_files}.pkl')

    # Confusion Matrix
    predictions = classifier.predict(dataset_valid)
    cm = confusion_matrix(labels_valid, predictions)
    print("Confusion Matrix:")
    print(cm)

    # Classification Report
    cr = classification_report(labels_valid, predictions)
    print("Classification Report:")
    print(cr)
    
    labels_names = ['dislike', 'fist', 'like', 'palm', 'peace']

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_names)
    disp.plot()
    plt.show()