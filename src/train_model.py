import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib

df = pd.read_csv("data/products.csv")

#Uklanjam razmake prazna polja u imenu kolona
df.columns = df.columns.str.strip()

#Uklanjam _ iz imena kolone 'Product Code'
df.rename(columns={'_Product Code': 'Product Code'}, inplace=True)

#Brisemo nedostajuce vrednosti
df = df.dropna()

#Pretvaramo sve vrednosti u kolni u mala slova
df['Category Label'] = (df['Category Label'].astype(str).str.strip())

#Standardizujemo kolnu
df['Category Label'] = df['Category Label'].replace({
   "fridge": "Fridges",
   "CPUs": "CPU",
   "Mobile Phone": "Mobile Phones"
})

#Delimo podatke na ulazne i ciljanu promenljivu 
X = df['Product Title']
y = df['Category Label']

pipeline = Pipeline([
      ("tfidf", TfidfVectorizer()),
      ("Classifier", LinearSVC())
])

#Treniramo model nad svim podacima
pipeline.fit(X, y)

#Cuvamo model u fajl
joblib.dump(pipeline, "model/category_model.pkl")

print("Model trained and saved as 'model/category_model.pkl'")