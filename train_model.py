import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier


FOLDERS = [
    "/home/nikodem/IVSPA/tmp_ready",
    "/home/nikodem/IVSPA/dislike_ready",
    "/home/nikodem/IVSPA/fist_ready",
]

classifierPath = f'Classifiers/'

if not os.path.exists(classifierPath):
    os.mkdir(classifierPath)

dataset = []
labels = []

for dataPath in FOLDERS:
    
    label_name = dataPath.removesuffix('_ready').split("/")[-1]
    print(labels)

    # read training file
    for file in os.listdir(dataPath):
        with open(f"{dataPath}/{file}", 'r') as f:
            data = f.read().removesuffix(";").split(';')
            data = np.array(data).astype(np.float32)
            dataset.append(data)
        labels.append(label_name)

print(labels)

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, stratify=labels)



classifier = RandomForestClassifier()

classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print(score)
joblib.dump(classifier, f'test.pkl')
