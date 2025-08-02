# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 14:38:45 2025

@author: ozgeb
"""

# %% veri yukleme ve on isleme

import sys
print(sys.executable)



import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

digits = load_digits()
X,y = digits.data, digits.target



plt.figure(figsize=(8, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(digits.target[i])
    plt.axis('off')
plt.suptitle("İlk 10 Rakam Görseli", fontsize=14)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# veriyi standaridze hale getir

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# %% pca ile boyut indirgeme (dimension reduction)

pca = PCA(n_components = 0.95)  # varyans %95'i koruyacak sekilde istedigin kadar component olustur sonucunda 64'ten 38 boyuta dustu
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# burada gorsellestirme icin 2 boyuta dusuruyoruz
pca_2d = PCA(n_components = 2)
X_train_pca_2d = pca_2d.fit_transform(X_train_scaled)
X_test_pca_2d = pca_2d.transform(X_test_scaled)

plt.figure(figsize = (20,20))

for i in np.unique(y_train):
    plt.scatter(X_train_pca_2d[y_train == i,0], X_train_pca_2d[y_train == i,1],
                label = digits.target_names[i])

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("PCA ile 2 boyutlu veri gorsellestirme")

plt.legend(title = "Siniflar", loc = "best")
plt.show()


# %% model training and grid search

# svm
svm_params = {"C":[0.1,1,10], "kernel": ["linear", "rbf"]}
svm = SVC()
svm_grid = GridSearchCV(svm, svm_params, cv = 5)
svm_grid.fit(X_train_pca, y_train)

# random forest
rf_params = {"n_estimators": [50,100,200]}
rf = RandomForestClassifier(random_state = 42)
rf_grid = GridSearchCV(rf, rf_params, cv = 5)
rf_grid.fit(X_train_pca, y_train)

# knn
knn_params = {"n_neighbors": [3,5,7]}
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, knn_params, cv = 5)
knn_grid.fit(X_train_pca, y_train)

# en iyi parametreye sahip modellerin secilmesi
best_svm = svm_grid.best_estimator_
best_rf = rf_grid.best_estimator_
best_knn = knn_grid.best_estimator_

# %% voting classifier

voting_classifier = VotingClassifier(estimators = [
    ("svm", best_svm),
    ("rf", best_rf),
    ("knn", best_knn)], voting = "hard")

voting_classifier.fit(X_train_pca, y_train)

y_pred = voting_classifier.predict(X_test_pca)

# %% sonuclarin degerlendirilmesi

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = digits.target_names)
disp.plot(cmap = plt.cm.Blues)
plt.show()

print(f"Best svm params: {svm_grid.best_params_}")
print(f"Best rf params: {rf_grid.best_params_}")
print(f"Best knn params: {knn_grid.best_params_}")
print(f"Voting classifier accuracy: {voting_classifier.score(X_test_pca, y_test)}")



# Best svm params: {'C': 1, 'kernel': 'rbf'}
# Best rf params: {'n_estimators': 100}
# Best knn params: {'n_neighbors': 3}
# Voting classifier accuracy: 0.9861111111111112













