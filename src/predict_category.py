import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load("model/category_model.pkl")

print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")

while True:
    title = input("Enter review title: ")
    if title.lower() == "exit":
        print("Exiting ...")
        break
    
    prediction = model.predict([title])[0]
    print(f"Predicted sentiment: {prediction}\n" + "-" * 40)