import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


df = pd.read_csv("otodom.csv")
#prawdziwe dane z 2021, pełne błędów, mało kolumn

#zmienna zależna - cena. Pozostałe to zestaw cech
print(df.describe())  #rozkład, czy niema ujemnych. Ceny będą nam brużdzić
#trzeba sprawdzićkorelacje między zmienną niezależną (5 cech) i korelacja między ceną, a poszczególnymi cechami.,
#

print(df.corr())
#z korelaci nic nie widać, dlateo seaborn

sns.heatmap( df.corr(), annot=True )
#lt.show()

#weźmy tylko niektóre dane, tylko w  ramach zmiennej niezależnej, wycinamy cenę
sns.heatmap( df.iloc[: , 2:].corr() , annot=True)
#plt.show()
#porządane, aby cenacha silnie korelowała ze zmeianą wynikową (z ceną), niepożądane koeracje między cehcami
#dobra korelacha między x, a y, a nie między Xami.  Ale.... powiechnia i liczba pokoi - ok.   Niewskazana współliniowość - ale drzewo decyzyjne to wykorzystuje jako booster
sns.displot(df.cena)  #zobaczmy ceny
#plt.show()  #dziwne, robienie regresji na tych danych bez sensu.
plt.scatter(df.powierzchnia, df.cena)
#plt.show()   #trzeba wybrać dane
print(df.describe())   #max 10 milionoów, średnia6 861tyś, mediana odstae dosć mocno, 667tyś.  Weźmy fragment między zakresami
#między min, a 75%

_min = df.describe().loc["min","cena"]
q1 = df.describe().loc["25%","cena"]
q3 = df.describe().loc["75%","cena"]
print(_min, q1, q3)
df1 = df[(df.cena>=_min)&(df.cena<=q3)]
sns.displot(df1.cena)
plt.show()  #sząłu nie ma, ale jest dużo lepiej, na tym zbiorze danych wykonamy estymator, żeby mieć predyktor.

#wcześniej tylko trenignowe,   teraz dane treningowe i dane testowe osobno.
X = df1.iloc[: , 2:]  #zmienna niezależna
y = df1.cena

#train_test_split() - dzieli dane losowo na dane treningowe i testowe

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)     #najlepiej 80/20, od random zależy pseudoosowość
#na wyjściu dostaniemy krotkę.
print(X_train.shape, X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))   #szybki test, zwraca współczynnik dopasowania. podajemy dane testowe. Walidujemy modek
#0.67. Słabo, ale też nie ogarnialismy danych dobrze

print(model.coef_)   #współczyniki powiązane z cechami naszej zmiennej niezależnej. Zbudujmy dataframe, do intdeksu wrzucamy kolumny
print(pd.DataFrame(model.coef_ , X.columns))

#zwiększenie liczby pięter podnosi wartość o 1063zł.  Zwiększenie liczby pokoi, zmniejsza cnę... a stoi w korelacji z powierzchnią.
#2 cechy, które ze sobą korelują, ale wychodzi dziwnie.  W uczeniu masyznowym trzebaby usunąć liczbę pokoi

#teraz mała zmiana może dużo zmienić.  Weźmy drzewo decyzyjne.
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
#jaka zwykła regresja jest słaba, beez parmetryzowania mamy dużo lepiej
#regresja liniowa, stara, ale do dobrychdanych.

#czasem trzeba wybrać dobre dane, żeby bylo lepiej. Drzewo decyzyjne samo umie wybrać te cenhy, ktróę są najważniejsze. Jak dzłowiek. Od ogłułu do szczegółu