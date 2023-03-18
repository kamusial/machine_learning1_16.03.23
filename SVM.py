#przykład, kulki i kij - granica. aby jak największą odległośc od watrtoscig ranicznych
#teraz jeden zbór w środku drugiego.  I tutaj SVM , w płaszczyźnie 3D
#maszyna wektóró nośnych- zrobić hiperpłąszczyznę, algorytm ma z 30 lat
#funkcja kernela, przekształcająca
#RDF - robi robotę

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

X, y = make_circles(500, factor=0.2, random_state=0, noise=0.6)
plt.scatter(X[:,0], X[:,1], c=y)
#liczba próbek
plt.show()

from sklearn.svm import SVC  #clasifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test) ) ))

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test))))

model = KNeighborsClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test))))

# 'linear', 'poly', 'rbf', 'sigmoid'
model = SVC(kernel='rbf')  #tylko radial base function, więcej wymiarów
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test) ) ))

#Wizualizacja SVM  (dla 2 klas najpirerw, żeby się nie prznikały - weźmy oddzielone)
df = pd.read_csv("iris.csv")
species = {
    "Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2
}
df["class_value"] = df["class"].map(species)
df = df.query(" class_value!=2 ")
plt.scatter(df.sepallength, df.sepalwidth, c=df.class_value)
plt.show()
#jak pokazać wymiarwanie?

X = df[ ["sepallength","sepalwidth"] ]
y = df.class_value

model = SVC()
model.fit(X, y)

vector = model.support_vectors_
print(vector)
#nanosimy na wykres
plt.scatter(df.sepallength, df.sepalwidth, c=df.class_value)
plt.scatter(vector[:,0], vector[:,1], c="r")
plt.show()
#punkty brzegowe sąsiadujące z prostą. Trzeba poprowadzic wekto.... linia regresji.
#SWN do klasyfikacji

