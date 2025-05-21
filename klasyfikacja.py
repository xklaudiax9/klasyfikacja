import pandas as pd
import numpy  as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder,RobustScaler
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report,make_scorer,roc_curve, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_regression, RFECV, f_classif, chi2
import warnings
warnings.filterwarnings('ignore')

baza = pd.read_csv('gym_members_exercise_tracking.csv', delimiter =',', decimal= '.')


print(baza.head())
baza.info()
null = baza.isnull().sum()
print("ilość missingów:\n{}".format(null))
print(round(baza.describe(),3))
tekst=baza.select_dtypes(include=['object']).columns
print(tekst)
nowe = baza.drop(columns = ['Gender','Workout_Type'])
int=baza.select_dtypes(exclude=['object']).columns
print(int)


indeks=[]
ilosc=[]
srednia=[]
odchylenie=[]
minimum=[]
maximum=[]
kwartyl1=[]
mediana=[]
kwartyl3=[]
ilosc_out=[]
udzial_out=[]
ilosc_out_z=[]
udzial_out_z=[]


for cecha in int:
    indeks.append(cecha)
    ilosc.append(nowe[cecha].count())
    srednia.append(nowe[cecha].mean())
    odchylenie.append(nowe[cecha].std())
    minimum.append(nowe[cecha].min())
    maximum.append(nowe[cecha].max())
    kwar1=nowe[cecha].quantile(0.25)
    kwar3=nowe[cecha].quantile(0.75)
    kwartyl1.append(kwar1)
    mediana.append(nowe[cecha].quantile(0.50))
    kwartyl3.append(kwar3)
    iqr=kwar3-kwar1
    ilosc_out.append(sum(nowe[cecha]>kwar3+1.5*iqr)+sum(nowe[cecha]<kwar1-1.5*iqr))
    udzial_out.append((sum(nowe[cecha]>kwar3+1.5*iqr)+sum(nowe[cecha]<kwar1-1.5*iqr))/(nowe[cecha].count())*100)
    out_z_score = (sum(((nowe[cecha]-nowe[cecha].mean())/(nowe[cecha].std()))>3)+
                      sum(((nowe[cecha]-nowe[cecha].mean())/(nowe[cecha].std()))<-3))
    ilosc_out_z.append(out_z_score)
    udzial_out_z.append(out_z_score/(nowe[cecha].count())*100)

tabela=pd.DataFrame({'Ilość':ilosc,"Średnia":srednia,"Odchylenie":odchylenie,"Minimum":minimum,"Maksimum":maximum,
                    "I kwartyl":kwartyl1,"Mediana":mediana,"III kwartyl":kwartyl3, "Outliers rozstęp":ilosc_out,
                     "Outliers Z-score":ilosc_out_z, "% outliersów ":udzial_out, "% outliersów Z-score":udzial_out_z},index=indeks)

print(tabela)

class TransformerWOE(TransformerMixin,BaseEstimator):
    
    def __init__(self, cechy):
        self.cechy=cechy

            
    def fit(self,X,y):
        X_new=pd.concat([X, y], axis=1)
        
        target_name=X_new.columns[len(X_new.columns)-1]
        
        self.lista_woe_mapping=[]
        for cecha in self.cechy:
            woe_mapping=dict()
            for x in X_new[cecha].unique():
                non_event=len(X_new[X_new[cecha]==x])-sum(X_new[X_new[cecha]==x][target_name])
                non_event_total=sum(y==0)
                non_event_perc=non_event/non_event_total           
                event=sum(X_new[X_new[cecha]==x][target_name])
                event_total=sum(y==1)
                event_perc=event/event_total
                woe=np.log(non_event_perc/event_perc)
                woe_mapping[x]=woe   
            self.lista_woe_mapping.append(woe_mapping)
        return self
            
    
    def transform(self,X):
        i=0
        for cecha in self.cechy:
            X[cecha]=X[cecha].map(self.lista_woe_mapping[i])
            i=i+1
        return X
    
    

y=nowe['Calories_Burned']
x=nowe.drop(columns = ['Calories_Burned'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2330)


model = LogisticRegression(max_iter= 1000,multi_class='ovr')
model.fit(x_train,y_train)
train_pred_y = model.predict_proba(x_train)
test_pred_y = model.predict_proba(x_test)
#print('y_train_klasy:',set(y_train))
#print('y_test_klasy',set(y_test))
def podział(x):
    if x<500:
        return 0
    elif x<1000:
        return 1
    else:
        return 2

nowe['klasa_kalorii'] = nowe['Calories_Burned'].apply(podział)
x = nowe.drop(columns =['Calories_Burned','klasa_kalorii'])
y = nowe['klasa_kalorii']
from sklearn.preprocessing import LabelEncoder

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2330)

model = LogisticRegression(max_iter= 1000,multi_class='ovr')
model.fit(x_train,y_train)
train_pred_y = model.predict_proba(x_train)
test_pred_y = model.predict_proba(x_test)
#AUC
train_auc = roc_auc_score(y_train,train_pred_y, multi_class= 'ovr')
test_auc = roc_auc_score(y_test,test_pred_y, multi_class= 'ovr')
#gini
train_gini = 2*train_auc - 1
test_gini = 2*test_auc - 1



Result=pd.DataFrame([{"AUC":train_auc,"GINI":train_gini}])
ResultTest=pd.DataFrame([{"AUC_test":test_auc,"GINI_test":test_gini}])
print('treningowe:',Result)
print('testowe:',ResultTest)

y_true_1 = [0 for i in range(90)]+[1 for i in range(10)]
y_pred_1 = [0 for i in range(99)]+[1 for i in range(1)]
prec = precision_score(y_true_1, y_pred_1)
recal = recall_score(y_true_1,y_pred_1)
acc = accuracy_score(y_true_1, y_pred_1)
print(f'precyzja = {prec}')
print(f'czułość = {recal}')
print(f'dokładność = {acc}')
sred_aryt = np.mean([prec,recal])
f1 = f1_score(y_true_1, y_pred_1)
print(f'średnia arytmetyczna = {sred_aryt}')
print(f'miara f1 = {f1}')
print(classification_report(y_true_1, y_pred_1))

nazwy= ["Regresja logistyczna", "Lasy losowe", "XGBoost", "Bagging (naiwny bayes)", "Bagging (drzewo decyzyjne)"]

modele=[LogisticRegression(), RandomForestClassifier(max_depth=5),
        XGBClassifier(verbosity = 0), BaggingClassifier(estimator =GaussianNB()),
        BaggingClassifier(DecisionTreeClassifier())]


accuracy_train = []
accuracy_test = []
auc_train = []
auc_test = []
f1_train = []
f1_test = []
pot_profit_train = []
pot_profit_test = []
false_positive = []
false_negative = []
true_positive = []
recall = []
precision = []

for model in modele:
    model.fit(x_train,y_train)
    accuracy_train.append(accuracy_score(y_pred=model.predict(x_train),y_true=y_train))
    accuracy_test.append(accuracy_score(y_pred=model.predict(x_test),y_true=y_test))
    auc_train.append(roc_auc_score(y_score=model.predict_proba(x_train),y_true=y_train,multi_class='ovr'))
    auc_test.append(roc_auc_score(y_score=model.predict_proba(x_test),y_true=y_test,multi_class='ovr'))
    f1_train.append(f1_score(y_pred=model.predict(x_train),y_true=y_train,zero_division=0,average='macro'))
    f1_test.append(f1_score(y_pred=model.predict(x_test),y_true=y_test,zero_division=0,average='macro'))
    
    FP=np.sum(np.logical_and(model.predict(x_test) == 1, y_test == 0))
    TP=np.sum(np.logical_and(model.predict(x_test) == 1, y_test == 1))
    FN=np.sum(np.logical_and(model.predict(x_test) == 0, y_test == 1))
    false_positive.append(FP)
    false_negative.append(FN)
    true_positive.append(TP)
    recall.append(TP/(TP+FN))
    precision.append(TP/(TP+FP))


podsumowanie=pd.DataFrame({"Acc_train":accuracy_train, "Acc_test":accuracy_test,
                           "AUC_train":auc_train, "AUC_test":auc_test,
                           "F1_train":f1_train, "F1_test":f1_test, 
                           "false_positive": false_positive,"false_negative": false_negative,"true_positive": true_positive,
                           "recall":recall, "precision":precision},index=nazwy)
print(podsumowanie)

# dla lasów losowych, XGBoost, Bagging(drzwo decyzyjne) występuje overfitting dlatego odrzucamy te metody z analizy
#Regresja logistyczna wysoki recall, model wychwytuje wszystkie pozytywne przypadki, delikatna nierównowaga precision/ recall ale generalnie wygląda dobrze
#Bagging(naiwny bayes) brak overfitingu ale metryki niższe niż w XGBoost, zbyt prostymodel

#w momencie zmiejszenia max_depth wyniki trochę się zmieniły: XGBoost nadalma mocny overfiting ale Lasy losowe już nie
