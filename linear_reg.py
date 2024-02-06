#import the librabry
import pandas as pd 
import matplotlib.pyplot as plt


score_data=pd.read_csv("C:/API/app/GAP_Prediction.csv")

#First n rows of the DataFrame
print(type(score_data))
score_data.head()

#ndex, Datatype and Memory information
score_data.info()

#Summary statistics for numerical columns
score_data.describe()

#check for null value 
score_data.isnull().sum()

#visualize the distirubtion
score_data.hist(bins=5)

#correlation between columns 
score_data.corr()

#return a numpy array 
X=score_data["SAT"].values.reshape(-1,1)
Y=score_data["GPA"].values.reshape(-1,1)


plt.scatter(X,Y,color='b')
plt.xlabel("SAT")
plt.ylabel("GPA")


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.25,random_state=42)

print("xtrain",xtrain.shape)
print("xtest",xtest.shape)
print("ytrain",xtrain.shape)
print("ytest",ytest.shape)


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
model=lr.fit(xtrain,ytrain)

ypred=model.predict(xtest)


plt.scatter(X,Y,color='r')
plt.plot(xtest,ypred)
#plt.xlim(100,)
#plt.ylim(0,)
plt.xlabel("SAT")
plt.ylabel("GPA")


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

print("MAE",mean_absolute_error(ytest,ypred))

print("MSE",mean_squared_error(ytest,ypred))

#Root mean Squared erro
from  math import sqrt as sqrt
print("RMSE",sqrt(mean_squared_error(ytest,ypred)))
r2= r2_score(ytest,ypred)
print("R^2 (coefficient of determination ,)",r2)
N=len(ytest)
p=1


print(model.coef_)
print(model.intercept_)


ar2=1-((1-r2)*(1-N))/(N-p-1)

print("Ajusted r2" , ar2)




