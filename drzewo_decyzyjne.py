import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("iris.csv")
df["class"].value_counts()
species = {
    "Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2
}
df["class_value"] = df["class"].map(species)

plt.figure(figsize=(7,7))
sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class' )

plt.figure(figsize=(7,7))
sns.scatterplot(data=df, x='petallength', y='petalwidth', hue='class' )

sns.heatmap( df.iloc[:, :4].corr(), annot=True)
X = df[ ["petallength","petalwidth"] ]
y = df.class_value

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, random_state=0)
model.fit(X, y)
#nie dzielimy na dane tetowe , bo chcemy tylko pokazać krawędzie decyzyjne
from mlxtend.plotting import plot_decision_regions

plt.figure(figsize=(7,7))
plot_decision_regions(X.values, y.values, model)


from dtreeplt import dtreeplt

dtree = dtreeplt(model=model, feature_names=X.columns, target_names=["setosa","versicolor","virginica"])
dtree.view()
plt.show()


X = df[ ["sepallength","sepalwidth"] ]
y = df.class_value

model = DecisionTreeClassifier(max_depth=5, random_state=0)
model.fit(X, y)
plt.figure(figsize=(7,7))
plot_decision_regions(X.values, y.values, model)
dtreeplt(model, feature_names=X.columns, target_names=["setosa","versicolor","virginica"]).view()
plt.show()

#krawędzie decyzyjne
# max dept 5 - niewysarczające zmienić na 9
# nie daje rady, zabrzudzony zbiór danych - to nie powinno mieć miejsca. NIe są to linie
# drzewko jest rpozeuczone.. łatwo wpada w przeuczanie

# estymator dla 4 cech

X = df.iloc[: , :4]
y = df.class_value

model = DecisionTreeClassifier(max_depth=9, random_state=0)
model.fit(X, y)

dtreeplt(model, feature_names=X.columns, target_names=["setosa","versicolor","virginica"]).view()
plt.show()
#sepallength - nie używane.   sepallength","sepalwidth" - nie są skorelowane, drzewo je omija
print(pd.DataFrame( model.feature_importances_ , X.columns))
#jak sięnie narobić, a zarobić, drapieżnik
#opiera się na badaniu wskaźnika gini i entropii
#max_features - ile max cech można wziąć. Hiperparametryzacja.
#min samples leave

