import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import  sklearn.preprocessing as prep
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

df = pd.read_csv('C:/Users/student/Downloads/agaricus-lepiota.data')
num_col = df.shape[1] #Broj atributa koje ima tabela sa pecurkama.
features = df.columns[1:num_col].tolist() #Svi atributi koji nisu atribut klase (ciljni atribut).

x_og_set = df[features] #Skup atributa na osnovu koga dodeljujemo klase.
#U opisu skupa podataka su navedeni atributi i znacenja njihovih vrednosti, prvi atribut je jestivost i otrovnost,
#jer su vrednosti e i p.
y_set = df[df.columns[0]] #Atribut klase.
num_classes = len(y_set.unique())

#Preprocesiranje: podatke obradjujemo tako da budu pogodni za metode klasifikacije.
#Ne mogu da se standardizu podaci bez dodatne obrade, jer podaci nisu numericki, nego imenski.
#Potrebno je enkodirati podatke da bi mogli da se normalizuju i kasnije obrade.

#Enkodiranje labela, jer su podaci stringovi. U petlji se enkodira svaka kolona tabele, jer metoda LabelEncoder
#kao argument prima samo jednidimenzioni niz.

le = prep.LabelEncoder()
for i in range(0,num_col-1):
    #Izlazi warning da se menja kopija a ne originalni objekat. Taj warning je lazno pozitivan, tj. on kaze da postoji
    #sansa da se tako nesto desi, ali u ovom slucaju se ne desava, pa treba da se podesi chained_assignment na none
    #sto oznacava da se sve iteracije petlje izvrsavaju od jednom.
    pd.options.mode.chained_assignment = None
    x_og_set[x_og_set.columns[i]] = le.fit_transform(x_og_set[x_og_set.columns[i]])

x_og_set.to_csv('preprocesirani_podaci.csv')

#Pravljenje objekata DataFrame.
X = pd.DataFrame(x_og_set)
y = pd.DataFrame(le.fit_transform(y_set))


#Normalizacija numerickih podataka.
scaler = prep.MinMaxScaler().fit(X)
X =pd.DataFrame(scaler.fit_transform(X))
X.columns = features

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y_set)

#Posto su svi atributi imenski, i kodirani su brojevima, ne smemo da koristima algoritme koji racunaju rastojanje
#medju slogovima, jer ne moze da se racuna rastojanje medju imenskim podacima. Zato ne moze da se primenjuje
#algoritam k susednih, ili Naivni Bajes sa Gausovom raspodelom, jer ne mozemo da pretpostavimo Gausovu raspodelu nad
#kodiranim podacima. Koriscenje metode su Stablo odlucivanja, Random Forest Klasifikaciju i Naivnu Bajesovu
#klasifikaciju sa pretpostavkom o uniformnoj raspodeli kodiranih vrednosti i sa naucenom raspodelom kodiranih vrednosti.

#Osim sto veceg procenta tacnosti metode, posebno nam je potrebno da bude sto manji procenat lazno tacnih vrednosti
#(pecurke koje su otrovne a klasifikovane su kao jestive).

#Stablo odlucuvanja:
clf_DTC = DecisionTreeClassifier(criterion='gini')

start_DTC=time.time()
clf_DTC.fit(X_train,y_train)
end_DTC = time.time()
vreme_DTC = end_DTC-start_DTC

print(clf_DTC.score(X_train,y_train))
print(clf_DTC.score(X_test,y_test))
print("Vreme za koje je izvrsen metod Stabla Odlucivanja je ")
print(vreme_DTC)

y_train_pred = clf_DTC.predict(X_train)
y_test_pred = clf_DTC.predict(X_test)

conf_DTC = metrics.confusion_matrix(y_test, y_test_pred)
print('Matrica konfuzije za Stablo Odlucivanja:\n')
print(conf_DTC)

plt.imshow(conf_DTC)
plt.colorbar()
plt.xticks(range(num_classes), y_set.unique())
plt.yticks(range(num_classes), y_set.unique())
plt.title('Matrica konfuzije - Decision Tree Classifier')
plt.show()

cnf_matrix = metrics.confusion_matrix(y_test, y_test_pred)

clf_RFC = RandomForestClassifier(n_estimators=3, max_depth=40)
start_RFC=time.time()
clf_RFC.fit(X_train,np.ravel(y_train))
end_RFC = time.time()
vreme_RFC = end_RFC-start_RFC

print(clf_RFC.score(X_train,y_train))
print(clf_RFC.score(X_test,y_test))
print("Vreme za koje je izvrsen metod Slucajne Sume je")
print(vreme_RFC)

y_train_pred2 = clf_RFC.predict(X_train)
y_test_pred2 = clf_RFC.predict(X_test)

conf_RFC = metrics.confusion_matrix(y_test, y_test_pred2)
print('Matrica konfuzije za Slucajnu Sumu:')
print(conf_RFC)

plt.imshow(conf_RFC)
plt.colorbar()
plt.xticks(range(num_classes), y_set.unique())
plt.yticks(range(num_classes), y_set.unique())
plt.title('Matrica konfuzije - Random Forest Classifier')
plt.show()

cnf_matrix2 = metrics.confusion_matrix(y_test, y_test_pred2)

clf_MNB_unif = MultinomialNB(fit_prior=False)
start_MNB_unif=time.time()
clf_MNB_unif.fit(X_train,np.ravel(y_train))
end_MNB_unif = time.time()
vreme_MNB_unif = end_MNB_unif-start_MNB_unif

print(clf_MNB_unif.score(X_train,y_train))
print(clf_MNB_unif.score(X_test,y_test))
print("Vreme za koje je izvrsen Multionomijani\nNaivni Bajesov algoritam sa unif. raspodelom je ")
print(vreme_MNB_unif)

y_train_pred3 = clf_MNB_unif.predict(X_train)
y_test_pred3 = clf_MNB_unif.predict(X_test)

conf_MNB_unif = metrics.confusion_matrix(y_test, y_test_pred3)
print('Matrica konfuzije za Multinomialni Naivni Bajesov\nalgoritam sa pretpostavkom o uniformnoj raspodeli:')
print(conf_MNB_unif)

plt.imshow(conf_MNB_unif)
plt.colorbar()
plt.xticks(range(num_classes), y_set.unique())
plt.yticks(range(num_classes), y_set.unique())
plt.title('Matrica konfuzije - Multinomial Naive Bayes\nsa pretpostavkom o uniformnoj raspodeli:')
plt.show()

cnf_matrix3 = metrics.confusion_matrix(y_test, y_test_pred3)

clf_MNB_fp = MultinomialNB(fit_prior=True)

start_MNB_fp=time.time()
clf_MNB_fp.fit(X_train,np.ravel(y_train))
end_MNB_fp = time.time()
vreme_MNB_fp = end_MNB_fp-start_MNB_fp

print(clf_MNB_fp.score(X_train,y_train))
print(clf_MNB_fp.score(X_test,y_test))
print("Vreme za koje je izvrsen Multionomijani\nNaivni Bajesov algoritam sa naucenom raspodelom je ")
print(vreme_MNB_fp)
y_train_pred4 = clf_MNB_fp.predict(X_train)
y_test_pred4 = clf_MNB_fp.predict(X_test)

conf_MNB_fp = metrics.confusion_matrix(y_test, y_test_pred4)
print('Matrica konfuzije za Multinomialni Naivni Bajesov\nalgoritam sa prethodno naucenom raspodelom:')
print(conf_MNB_fp)

plt.imshow(conf_MNB_fp)
plt.colorbar()
plt.xticks(range(num_classes), y_set.unique())
plt.yticks(range(num_classes), y_set.unique())
plt.title('Matrica konfuzije - Multinomial Naive Bayes\nsa prethodno naucenom raspodelom')
plt.show()

cnf_matrix4 = metrics.confusion_matrix(y_test, y_test_pred4)

min = vreme_DTC
min_name = "DTC"
if vreme_MNB_unif < min:
    min = vreme_MNB_unif
    min_name = "MNB_unif"
if vreme_RFC < min:
    min = vreme_RFC
    min_name = "RFC"
if vreme_MNB_fp < min:
    min = vreme_MNB_fp
    min_name = "MNB_fp"

print("Najkrace vreme izvrsavanja je za "+ min_name+".")
