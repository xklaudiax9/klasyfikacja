
#zadanie łatwe
import pandas as pd;
import numpy as np;
baza = pd.read_csv("gym_members_exercise_tracking.csv", sep = ',')
print(baza.head())

baza.info()
nowe = baza.drop(columns = ['Gender','Workout_Type'])

stati = nowe.describe()
print(round(stati,2))

m = round(nowe.median(),2)
s = round(nowe.mean(),2)
o = round(nowe.std(),2)

a = pd.DataFrame(columns = [])
a.insert( loc = 0,column =  'mediana',value = m )
a.insert( loc = 0,column =  'średnia',value = s )
a.insert( loc = 0,column =  'odchylenie',value = o )

# wynik zadania łatwego
print(a)



#zadanie trudne
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

baza = pd.read_csv("gym_members_exercise_tracking.csv", sep = ',')
nowe = baza.drop(columns = ['Gender','Workout_Type'])

X = nowe.drop(columns=['Calories_Burned'])
y = nowe['Calories_Burned']

model = LinearRegression()
model.fit(X,y)
mse = round(mean_squared_error(y,model.predict(X)),2)
r2 = round(model.score(X,y),4)
mea = round(mean_absolute_error(y,model.predict(X)),2)
#wynik zadania trudnego
print('1) MSE:','  R^2:','  MAE:')
print(mse ,r2,mea)
# R^2: model świetnie objaśnia zmienną objaśniającą w 97,22%
v = round((np.std(y)/np.mean(y)),3)
print(v)
# MSE: model może się mylić o 2060,66 jednostki, jest to stosunkowy duży błąd 
# ale akceptowalny poniewaz wspólczynnik zmiennosci zmiennej objasnianej wynosi 30% co wskazuje na wysoka zmiennosc zmiennej
# MAE: wynik może się różnić średnio o 35,4 jednostki od rzeczywistej wartości

#poprawa próba innego modelu (jednak nie)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
trainX, testX,trainY,testY = train_test_split(X,y,test_size=0.2,random_state=30)
ml = RandomForestRegressor(n_estimators = 150, random_state= 30)
ml.fit(trainX,trainY)
mse2 = round(mean_squared_error(testY,ml.predict(testX)),2)
r22 = round(r2_score(testY,ml.predict(testX)),4)
mae2 = round(mean_absolute_error(testY,ml.predict(testX)),2)
print('2) MSE:',"  R^2:",'  MAE:')
print(mse2,r22,mae2)
#eee jednak wszystko pogorszyło, lepiej wykorzystać klasyczną regresję liniową
