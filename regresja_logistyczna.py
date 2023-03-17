import pandas as pd
import numpy as np

df = pd.read_csv("diabetes.csv")
print(df)
#diabetis function, kwestie związane z obciążeniami genetycznymi, ważne żeby sierozeznacać w kolimnach
#wartości 0 - coś źżle
print(df.describe())
print(df.describe().to_string())
#wartości zerowe chcemy zastąpić średnimi (bez zer). Dla ćwiczenia. Count - widać, że gdzieniegdzie nie ma wartości
print(df.isna().sum())
#trzeba przyjąć jakąś taktykę....

#ktore kolumny.. vez pregnancy i outcome

print(df.outcome.value_counts())

print(df.columns)

for col in ['glucose', 'bloodpressure', 'skinthickness', 'insulin',
       'bmi', 'diabetespedigreefunction', 'age']:
    df[col].replace(0, np.NaN, inplace=True)
    mean_ = df[col].mean()
    df[col].replace(np.NaN, mean_, inplace=True)

#mean nie bierze śrdniej. Zera konnwertujemy na NaN, a potem na mean
#replace zwróciłby kopię

print(df.isna().sum())
#wysokie BMI i glukoza

print(df.describe())

df.to_csv("cukrzyca.csv", index=False)
#żeby nie zginęło

#nowa metryka - legresja logicztyczna to klasyfikaja

X = df.iloc[: , :-1]  #wszystkie wiersze i mamy wszystkie kolumny bez ostatniej
y = df.outcome
#print(X)
from sklearn.linear_model import LogisticRegression   #regresja jest tu
from sklearn.model_selection import train_test_split   #klasa jest tu
from sklearn.metrics import confusion_matrix    #metuda macież pomyłek / konfuzji
#lepiej się pomylić na korzyść klienta, czy banku

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#random state będzie nas rpześladował

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
#potrzebna metryka, jak się myli.... na konrzyść której klasy.  w 25% model się myli
#niezgodnosć miedzy danymi predykcyjnymi, a rzeczywistymi

#predict - znamy
#musimy skonfrontować wartości y testowe, z wartościami obliczonymi na pdostaiw X testowych
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test) ) ))

#pionowo, klasy rzeczywiste, poziomo, wartości predyktowane

#spróbujemy zbalansować outcome

print('Zmiana danych')
print(df.outcome.value_counts())
df1 = df.query(" outcome==0 ").sample(n=500, random_state=0)
df2 = df.query(" outcome==1 ").sample(n=500, random_state=0)
#obekty data frame mają metody sample
df3 =pd.concat([df1,df2])
X = df3.iloc[: , :-1]  #wszystkie wiersze i mamy wszystkie kolumny bez ostatniej
y = df3.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test) ) ))
#zmniejszyła się liczba pomyłke w klasie 1.
#balansowanie klasą wyunikową wpływa na działanie estymatora
#dane testowe na 0.1

#%matplotlib inline   - silnik matplotlib, wszystkie wykresy będą renderowane w kodzie
#nie zadziała w pycharmie.


#możę można zostawićkod u nas i udstępnić tylko wynik. Model w chmurze. Można odpytać
#popularny model
#i jak utrwalić