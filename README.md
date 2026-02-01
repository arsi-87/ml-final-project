# Predikcija kategorija

Ovaj projekat se bavi problemom **predikcije kategorija (klasifikacije)** koristeci masinsko ucenje. Cilj je trenirati model koji
na osnovu ulaznih podataka, predvidja kojoj kategoriji pripada dati uzorak.

## 📌 Opis problema

Dat je skup podataka koji sadrzi:
- ulazne karakteristike 
- ciljnu varijablu 

Zadatak modela je da nauci obrazac iz podataka i pravilno klasifikuje nove, nepoznate primere.


## 📂 Struktura projekta

├── data/
│ ├── products.csv
│
├── src/
│ ├── predict_category.py # Skript za testiranje modela
│ ├── train_model.py # Script za treniranje i cuvanje modela
│
├── model/
│ ├── category_model.pkl
│
├── notebook/
│   ├── AS_ML_Final_project.ipynb # EDA i procesiranje
│
└── README.md


## ⚙️ Koriscene biblioteke

- pandas      
- scikit-learn  
- matplotlib.pyplot
- seaborn
- joblib


## 🧹 Priprema podataka

1. Pocetna podesavanja:

- Kreiran novi repozitorijum na GitHub-u
- Definisana struktura projekta
- Dodati podaaci

2. Analiza podataka:

- Ucitan i analiziran skup podataka sa proizvodima
- Koriscen matplotlib i seaborn za vizualizaciju
- Analiza i vizualizacija kolone 'Category Label'

3. Ciscenje podataka i predprocesiranje

- Uklanjanje nedostajucih vrednosti
- standardizacija kolone 'Category Label'
- konvertovan tip kolone 'Listing Date' u datetime

4. Inzinjering karakteristika:

- Odabir najvaznijih ulaznih karakteristika
- Brisanje nebitnih kolona
 

## 🧠 Treniranje modela i Evaluacija

Model se trenira koriscenjem algoritama za klasifikaciju, u nasem slucaju su to:

- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- Naive Bayes  
- Decision Tree

Za svaki model:

 Podacai proci kroz pipline i definisu se koraci obrade podataka.
 
- Najpre se podatci transformisu
- Koristi se algoritam koji na osnovu tih brojeva uci sablon
- Na kraju,dobijeni model se koristi za pravljenje predikcija


📊 Evaluacija modela

Performanse modela merimo pomocu:

- tacnosti (accuracy)
- Klasifikacioni izvestaj:
	- precision
	- recall
	- F1- score po kategorijama
- matrica zabune

## Finalno treniranje modela

- Treniramo model na ceo skup podataka
- Cuvamo pipline pomocu joblib kao category_model.pkl

## Upotreba modela

- Ucitavamo sacuvanji model 
- Pomocu skripta 'predict_category' unosom teksta testiramo model na novim podacima

## Koriscenje modela: 

Pokrenmo u python-u train_model.py
U folderu 'model' se kreira fajl category_model.pkl
Pomocu interaktivnog skripta 'category_predict' predvidjamo kategorije nad  novim podacima.
