import pickle 
model=pickle.load(open("models/fake_news_model.pkl","rb"))
vectorizer=pickle.load(open("models/tfidf_vectorizer.pkl","rb"))

def classify_news(text):
    transform=vectorizer.transform([text])
    result=model.predict(transform)[0]
    return "Real" if result==1 else "false"

while True:
    user_input=input("Enter text:")
    print("prediction:",classify_news(user_input))

                  