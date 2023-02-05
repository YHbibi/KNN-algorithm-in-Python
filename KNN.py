import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


animals = pd.read_table('./data.txt')


# create a mapping from animal label value to animal name to make results easier to interpret

lookup_animal_name = dict(zip(animals.animal_label.unique(), animals.animal_name.unique()))   


# For this example, we use the mass, width, and height features of each animal instance

X = animals[['mass', 'width', 'height','age']].values
y = animals['animal_label']

# default is 75% / 25% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


#Create classifier object

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 4) # k = 4
knn.fit(X_train,y_train)
knn.score(X_test,y_test)




# first example: an animal with mass 20kg, width 1.3 m, height 0.5 m, age 10
animal_prediction = knn.predict([[20, 1.3, 0.5, 10]])
print(lookup_animal_name[animal_prediction[0]])

# second example: an animal with mass 4100kg, width 6.3 m, height 8.5 m, age 28
animal_prediction = knn.predict([[4100, 6.3, 8.5, 28]])
print(lookup_animal_name[animal_prediction[0]])

# third example: an animal with mass 200kg, width 6.3 m, height 8.5 m, age 13
animal_prediction = knn.predict([[200, 6.3, 8.5, 13]])
print(lookup_animal_name[animal_prediction[0]])


# chien if k <= 4 else non chat
animal_prediction = knn.predict([[9.3, 0.915, 0.51, 3.5]])
print(lookup_animal_name[animal_prediction[0]])
animal_prediction = knn.predict([[9.3, 0.915, 0.51, 3.5]])
print(lookup_animal_name[animal_prediction[0]])

