#import libraries
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#import dataset
df_fake=pd.read_csv("data/Fake.csv")
df_true=pd.read_csv("data/True.csv")
#labeling 
df_fake['label']=0
df_true['label']=1
#combine the datasets 
df=pd.concat([df_fake,df_true],axis=0)
print(df)
#independent and dependent variable
X=df["text"]
y=df["label"]
vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)
X=vectorizer.fit_transform(X)
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)


#save the trained model
import pickle
with open("models/fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

