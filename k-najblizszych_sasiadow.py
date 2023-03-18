#rozglądamy się wśród sąsiadów i ustalamy czego jest więcej.
#prosty algorytm, wolny - niedoskonały. Gdy przychodzi nowy punkt pomiarowy i trzeba od nowa przeglądać zbiór próbek
#odległość - najczęściej euklidesowej
#jak ocenić ten promień, sąsiedztwo.... k = ile próbek.... czasem nie trzeba za dużo
#przykłąd, godzina - wystarczy spytać osobę. Jedna soba może okłąmać. 3 osoby - mamy odpowiedź
# za mały K  - szumy
#jak punkty przemieszane, to k = 4 i 5 będą różne przypisania
#da siezrobić świetnie, albo słąbo, może różnie działać w zależności od harakterystyki punktóww

#3 garukni irysów. 2 rodzajepłatkó
#rozgladamy się po okolicy i do kogo jestemsy najbardzije podobni

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("iris.csv")
#nie można df.class
print(df["class"].value_counts())
#class - zmienn wynikowa, powinna mieć harakter numeryczny.
species = {
    "Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2
}
df["class_value"] = df["class"].map(species)
#chcemy nową kolumnę
#można z replace, ale wolimy map.
print(df["class_value"].value_counts())

#nowy kwiat z wbiorze
sample = np.array([5.6, 3.2, 5.2, 1.45])
#jak pokazać 4 cechy - zróbmy dekompozycje....

sns.scatterplot(data=df, x='sepallength', y='sepalwidth')
#plt.show()
#z tego wyjresu nic nie jest jasne

plt.figure(figsize=(7,7))
sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class')
#plt.show()

#nanieśmy dane naszego kwiatka.
plt.figure(figsize=(7,7))
sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class')
plt.scatter(5.6 , 3.2, c="r")
#plt.show()
#to może byc wszystko, zależy ile sąsiadów weźmiemy

plt.figure(figsize=(7,7))
sns.scatterplot(data=df, x='petallength', y='petalwidth', hue='class' )
plt.scatter(5.2 , 1.45, c="r")
plt.show()

# obliczać dystans pomiędzy próbką a istniejącymi danymi
df["distance"] = (df.sepallength-sample[0])**2 + (df.sepalwidth-sample[1])**2 +\
                 (df.petallength-sample[2])**2 + (df.petalwidth-sample[3])**2

print(df.sort_values("distance"))
print(df.sort_values("distance").head(3)["class"].value_counts())

#różne wyniki, specyfika algorytmu

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
print(df.head(5).to_string())
X = df.iloc[:, :4]
y = df.class_value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
#algorytm najbliższych sąsiadów w wersji klasyfikatora sklearn
model = KNeighborsClassifier(5)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
#każdy punkt ma taką samą wagę. Można zmienić
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test)  ) ))

#sprawdźmy
#model.prodict(sample) - błąd. Strutira 1dno wymiarowa
model.predict(sample.reshape(1,-1)) #klasyfikator wielokategoryczny, który musi na wyjściu dać wartość dyskretną
print(model.predict_proba(sample.reshape(1,-1))) #musi być struktura 2 wymiarowa

# jak wartość "k" wpływa na jakosc estymatora
result = []
for k in range(1,101):
    model = KNeighborsClassifier(k)
    model.fit(X_train, y_train)
    result.append( model.score(X_test, y_test) )

plt.plot( range(1,101), result)
plt.grid()
plt.show()
#zaleta, niewrażliwosć na warości odstające
#k - domyślnie 5

import joblib
model = KNeighborsClassifier(5)
model.fit(X_train, y_train)
model.score(X_test, y_test)
joblib.dump(model, "knn.model")
model1 = joblib.load("knn.model")
model1.predict(sample.reshape(1,-1))
dir(model1)
getattr(model1, 'n_neighbors')
print(model1.n_neighbors)

#i można leieć do końća z 02-knn

Porównanie z regresją logistyczną