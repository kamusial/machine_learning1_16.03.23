import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression



df = pd.read_csv('weight-height.csv',sep=',')
print(df.head(10))
df.Gender.value_counts()   #klasy zbalansowane
df.Height *= 2.54
df.Weight /= 2.2
#niezależne 2 kolumny - gender i height. Wynik weight. Szukamy ile powinna wyniesć masa ciała
#rozebrać zbiór wartości na zmienną niezależną i zależną. Bez wypisywania x i y,
#tylko wycinająć z data frame kolumn. Dobrze zobaczyć, jak wygląda rozkład wartości.

#Weight, zmienan zależna. Zobaczmy, czy da się zobrazować, jak rozkłąd normalny
#można użyć hist z pyplota. Wolimy teraz użyć seaborna

#sns.displot(df.Weight) #źle, bo razem dane dla kobiet i dla mężczyzn.
# sns.displot(df.query("Gender=='Male'").Weight)
# plt.show()
# sns.displot(df.query("Gender=='Female'").Weight)
# plt.show()

#problem z data frame. ZMienne niezależne.  Gender - wartości nienumeryczne - konwersja
#metoda replace, można inaczej. Get.dummies - zamienia na indykatory 0,1.
df = pd.get_dummies(df)
print(df)
del(df["Gender_Male"])
# #del w pythonie bardzo szerokie zastosowanie - usuwa co chcemy
print(df)
df.rename(columns={"Gender_Female":"Gender"}, inplace=True)   #bez kopii
print(df)
#dane na stole

#zmienna niezależna, wzrost, i płeć.
#dla Gender - 0 mezczyzna, 1-kobieta
#trzeba zaimportować odpowiedią klasę - linear regression
#można go sprytniewytresować. Nasz X to gender i heifgt.
model = LinearRegression()
model.fit( df[ ["Height","Gender"] ] , df["Weight"] )
#jak chcemy wyciągnac kolumny dajemy nawiasy, a wnich kolejne nawiasy, bo tam info nazach
print(model.coef_, model.intercept_)   # wartości, pierwszy oto Height, 2go to Gender.
df2 = pd.DataFrame(model.coef_, ["Height","Gender"])   #na peirwszym miejscu dalen, an drugim index
print(df2)

#intercept - wyraz wolny,  coef - współczynnik

#w jaki sposób znaleźć wzór, opisujący idealną wagę

#1. Własna formuła
gender = 0
height = 192
weight = model.intercept_ + 1.069*height -8.8*gender
print(weight)

#2. Predict
#model.predict([192, 0])  #źle, struktura jest płąska
print(model.predict([[192, 0], [167, 1]]))

