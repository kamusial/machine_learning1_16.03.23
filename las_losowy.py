import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#las, ewolucja drzew
#uczenie amszynowe przyzwyczaja się do daych.  CV pomaga. I tak samo las
#zrównoleglenie operacji, każde drzewo wykonuje prace niezależnoe. Różne zbiory danych.
#promy kosmiczne

df = pd.read_csv("iris.csv")
print(df["class"].value_counts())

species = {
    "Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2
}
df["class_value"] = df["class"].map(species)
print(df)

plt.figure(figsize=(7,7))
sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class' )

plt.figure(figsize=(7,7))
sns.scatterplot(data=df, x='petallength', y='petalwidth', hue='class' )

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X = df[ ["sepallength", "sepalwidth"] ]   #najpierw te kontrowersyjne
#X = df[ ["petallength", "petalwidth"] ]
y = df.class_value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test)) ))

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

plt.figure(figsize=(7,7))
plot_decision_regions(X_train.values, y_train.values, model)
plt.show()
#las, to po prostu ekipa drzew. Pojedyncze drzewo może działać lepiej. A tu głosowanie, szum

#Konfrontacja z regresją logistyczną
df = pd.read_csv("cukrzyca.csv")

X = df.iloc[: , :-1]
y = df.outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=0)
#potem zamiast 100 spróbować 30 i max depth do 11
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test)) ))
#zastosowanie - duża liczba cech, dużo danych - miliony. Mało danych - las nie pomoze, może zaszkodzić
