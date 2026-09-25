import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Prepare the dataset
# This step might take a while depending on the number of images in the dataset

input_directory = os.path.join(os.getcwd(), 'images/')
categories = [ 'empty', 'not_empty' ]

data = []
labels = []

print('Preparing the dataset...')

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_directory, category)):
        img_path = os.path.join(input_directory, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)
        print('Processed image: ', img_path)

data = np.asarray(data)
labels = np.asarray(labels)

# Train data

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# train classifier
param_grid = [ { 'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000] } ]

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(x_train, y_train)

# Test performance
best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)

print(f'{score * 100}% of samples were correctly classified')
print(f'Best parameters: {grid_search.best_params_}')

pickle.dump(best_estimator, open('model.pkl', 'wb'))