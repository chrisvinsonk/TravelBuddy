import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer

df = pd.read_csv("C:/Users/chris/Documents/NHCE/Second Year/Semester 4/Mini "
                 "Project/Project/tripadvisor_hotel_reviews.csv")
df
df.info()
df.shape
x = df['Review'].values
y = df['Rating'].values
y = y.reshape(-1, 1)
y
binarizer = preprocessing.Binarizer(threshold=2.1)
binarizer
binarizer.fit(y)
y_bin = binarizer.transform(y)
y_bin
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y_bin, random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(stop_words='english')
x_train_vect = vect.fit_transform(x_train)
x_test_vect = vect.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train_vect, y_train)
y_pred1 = model.predict(x_test_vect)
from sklearn.metrics import accuracy_score

accuracy_score(y_pred1, y_test)
from sklearn.svm import SVC

model1 = SVC()
model1.fit(x_train_vect, y_train)
y_pred1 = model1.predict(x_test_vect)
from sklearn.metrics import accuracy_score

accuracy_score(y_pred1, y_test)
from sklearn.pipeline import make_pipeline

model2 = make_pipeline(CountVectorizer(), SVC())  # create pipeline combining the CountVecorizer and SVM models into one
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
y_pred2
accuracy_score(y_pred2, y_test)
