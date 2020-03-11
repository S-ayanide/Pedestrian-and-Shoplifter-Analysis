# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 21:40:14 2020

J.A.R.V.I.S Says Hello

@author: Sayan
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pickle
import cv2

DATADIR = "D:/Projects/AI/Minor Project"
CATEGORIES = ["Walking","Faces"]

IMG_SIZE = 90
X = []
y = []

training_data = []

def create_training_data(): 
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to Walking and Faces dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):                    
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)                
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
                plt.imshow(img_array, cmap ="gray")
                plt.show()
            except Exception as e:
                pass
            
create_training_data()

random.shuffle(training_data)

for features, labels in training_data:
    X.append(features)
    y.append(labels)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()