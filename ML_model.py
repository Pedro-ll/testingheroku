# Machine learning libraries and functions
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import pandas as pd
import pickle

print("\n")
print("----------//---------")

iris_bunch=load_iris() # bunch works like a dictionary, but it is not a dictionary

iris_df=pd.DataFrame(iris_bunch["data"],columns=iris_bunch["feature_names"])

target=iris_bunch["target"]

# Split the train and the test

X_train,X_test,y_train,y_test=train_test_split(iris_df,target,random_state=1,test_size=0.2,train_size=0.8)

# Let's create the model: LogisticRegression

log_model=LogisticRegression(max_iter=1000)

#Train the model
log_model.fit(X=X_train,y=y_train)

#Predictions 
print("\n")
print("Predictions from X_test")
print(log_model.predict(X_test))

#Evaluate model

print("\n")
print("Model accuracy")
print(log_model.score(X_test,y_test))
print("\n")

#Save the model

pkl_file="Logistic_model.pkl"

with open(pkl_file,"wb") as file:
    pickle.dump(log_model,file)

