import numpy as np
import pandas as  pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

train = pd.read_csv('train.csv') ##wczytanie danych
test = pd.read_csv('test.csv')

train['train_test'] = 1
test['train_test'] = 0 
test['Survived'] = np.NaN
all_data = pd.concat([train,test])                      #utworzenie tabeli zbiiorczej


print(train.info())                                     #info o danych

#print(train.describe())

df_num = train[['Age','SibSp','Parch','Fare']]
df_cat = train[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]
#___________________________________________________
Rozpoznanie danych



for i in df_num.columns:
    plt.hist(df_num[i])                               #tabele z wykresami ilości pasażerów na statku w zależności od cechy
    plt.title(i)
    #plt.show()

print(df_num.corr())
sns.heatmap(df_num.corr())                              #korelacje cech kategoryzujących
plt.show()

print(pd.pivot_table(train,index = 'Survived', values = ['Age','SibSp','Parch','Fare']))  #info o osobach uratowanych w zależności od zmiennych kategoryzujących


for i in df_cat.columns:
    sns.barplot(df_cat[i].value_counts().index,df_cat[i].value_counts()).set_title(i)
    plt.show()


print(pd.pivot_table(train, index = 'Survived', columns = 'Pclass', values = 'Ticket', aggfunc= 'count'))                           
print()
print(pd.pivot_table(train, index = 'Survived', columns = 'Sex', values = 'Ticket', aggfunc= 'count'))
print()
print(pd.pivot_table(train, index = 'Survived', columns = 'Embarked', values = 'Ticket', aggfunc= 'count'))


train['cabin_multiple'] =train.Cabin.apply(lambda x: 0 if pd.isna(x)else len(x.split(' ')))

print(train['cabin_multiple'].value_counts())

print(pd.pivot_table(train, index ='Survived',columns = 'cabin_multiple', values = 'Ticket',aggfunc = 'count'))

train['cabin_adv'] = train.Cabin.apply(lambda x:str(x)[0])

print(train.cabin_adv.value_counts())
print(pd.pivot_table(train, index = 'Survived', columns = 'cabin_adv', values = 'Name', aggfunc= 'count'))


train['numeric_ticket'] = train.Ticket.apply(lambda x:1 if x.isnumeric()else 0 )
train['ticket_letters'] = train.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0 )
print(train['numeric_ticket'].value_counts())

pd.set_option('max_rows',None)
print(train['ticket_letters'].value_counts())
print(pd.pivot_table(train,index = 'Survived',columns = 'numeric_ticket', values = 'Ticket', aggfunc = 'count'))
print()
train['name_title']= train.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
print(train['name_title'].value_counts())

#__________________________________________________
#tworzenie nowych zmiennych kategoryzujących

all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x)else 1)  
all_data['cabin_adv'] = all_data.Ticket.apply(lambda x:1 if x.isnumeric()else 0)
all_data['numeric_ticket'] =all_data.Ticket.apply(lambda x:''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())


#Wypełnianie pustych wartości dla wieku i ceny biletu

all_data.Age = all_data.Age.fillna(train.Age.mean())
all_data.Fare = all_data.Fare.fillna(train.Fare.mean())
#Usunięcie 2 wierszy z zestawu treningowego nie mającej wartości Embarked

all_data.dropna(subset = ['Embarked'],inplace = True)

#cena biletu w postaci logarytmu naturalnego
all_data['norm_fare'] = np.log(all_data.Fare+1)
all_data['norm_fare'].hist()

all_data.Pclass = all_data.Pclass.astype(str)

all_dummies = pd.get_dummies(all_data[['Pclass','Sex','Age','SibSp','Parch','norm_fare','Embarked','cabin_adv','cabin_multiple','numeric_ticket','name_title','train_test']])

#zmiana danych na liczbowe

scale = StandardScaler()
all_dummies_scaled = all_dummies.copy()
all_dummies_scaled[['Age','SibSp','Parch','norm_fare']] = scale.fit_transform(all_dummies_scaled[['Age','SibSp','Parch','norm_fare']])

X_train = all_dummies_scaled[all_dummies_scaled.train_test ==1].drop (['train_test'], axis =1)
X_test = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis = 1)
y_train = all_data[all_data.train_test == 1 ].Survived


#ocena i dobór najlepszego modelu
#_____________________________________________________

gnb = GaussianNB()
cv = cross_val_score(gnb, X_train,y_train,cv = 5)
print('GaussianNB',cv)
print(cv.mean())

lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X_train,y_train, cv =5 )
print('LogisticRegression',cv)
print(cv.mean())

dt = tree.DecisionTreeClassifier(random_state=1)
cv = cross_val_score(dt,X_train,y_train,cv = 5)
print('DecisionTreeClassifier',cv)
print(cv.mean())

knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train,y_train,cv = 5)
print ('KNeighborsClassifier',cv)
print (cv.mean())

rf = RandomForestClassifier(random_state=1)
cv = cross_val_score(rf,X_train,y_train,cv = 5)
print('RandomForestClassifier', cv)
print(cv.mean())

svc = SVC(probability = True)
cv = cross_val_score(svc,X_train,y_train,cv=5)
print('svc',cv)
print(cv.mean())

xgb = XGBClassifier(random_state = 1)
cv = cross_val_score(xgb,X_train,y_train)
print('XGBClassifier',cv)
print(cv.mean())


def clf_performance(classifier,model_name):
    print(model_name)
    print('Best Score:' +str(classifier.best_score_))
    print('Best Parameters:' +str(classifier.best_params_))

#Tuning modeli z wykorzystaniem parametrów
#_______________________________________________________


lr = LogisticRegression()
param_grid = {'max_iter': [2000],
              'penalty':['l1','l2'],
              'C':np.logspace(-4,4,20),
              'solver':['liblinear']}

clf_lr = GridSearchCV(lr, param_grid = param_grid,cv = 5,verbose = True, n_jobs = -1)
best_clf_lr = clf_lr.fit(X_train,y_train)
clf_performance(best_clf_lr,'Logistic Regression')

knn = KNeighborsClassifier()
param_grid = {'n_neighbors':[3,5,7,9],
              'weights' :['uniform','distance'],
              'algorithm': ['auto','ball_tree','kd_tree'],
              'p':[1,2]}
clf_knn = GridSearchCV(knn,param_grid= param_grid,cv = 5,verbose = True, n_jobs= -1)
best_clf_knn = clf_knn.fit(X_train,y_train)
clf_performance(best_clf_knn,'KNN')

svc = SVC(probability=True)
param_grid = [{'kernel':['rbf'],'gamma':[.1,.5,1],
               'C':[.1,1,10]},
              {'kernel':['linear'],'C':[.1,1,10]},
              {'kernel':['poly'],'degree':[2,3,4,5],'C':[.1,1,10]}]                                                       #odkomentować
clf_svc = GridSearchCV(svc,param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_svc = clf_svc.fit(X_train,y_train)
clf_performance(best_clf_svc,'SVC')

rf = RandomForestClassifier(random_state = 1)
param_grid = {'n_estimators':[100,500,1000],
              'bootstrap':[True,False],
              'max_depth':[3,5,10,20,50,75,100,None],
              'max_features':['auto','sqrt'],
              'min_samples_leaf':[1,2,4,10],
              'min_samples_split':[2,5,10]}
clf_rf_rnd = RandomizedSearchCV(rf,param_distributions = param_grid,n_iter = 100, cv = 5, verbose = True,n_jobs = -1)
best_clf_rf_rnd = clf_rf_rnd.fit(X_train,y_train)
clf_performance(best_clf_rf_rnd,'RandomForest')  

rf = RandomForestClassifier(random_state = 1)
param_grid = {'n_estimators': [400,450,500],
              'criterion' :['gini','entropy'],
              'bootstrap':[True],
              'max_depth':[15,20,25],
              'max_features':['sqrt'],
              'min_samples_leaf':[2,3],
              'min_samples_split':[2,3]}

clf_rf =GridSearchCV(rf , param_grid = param_grid, cv = 5 ,verbose = True,n_jobs = -1)
best_clf_rf = clf_rf.fit(X_train,y_train)
clf_performance(best_clf_rf,'Random Forest') 


xgb = XGBClassifier(random_state = 1)

param_grid = {
    'n_estimators':[500,550],
    'colsample_bytree':[0.75,0.8,0.85],
    'max_depth':[None],
    'reg_alpha':[1],
    'reg_lambda':[2,5,10],
    'subsample': [ 0.55,0.6],
    'learning_rate':[0.5],
    'gamma':[0.5,1],
    'min_child_weight':[0.01],
    'sampling_method':['uniform']
}

clf_xgb = GridSearchCV(xgb,param_grid = param_grid, cv = 5, verbose = True,n_jobs = -1)
best_clf_xgb = clf_xgb.fit(X_train,y_train)
clf_performance(best_clf_xgb,'XGB')


#predykcja z wykorzystaniem votingclassifier
#_________________________________________________
best_lr = best_clf_lr.best_estimator_
best_knn = best_clf_knn.best_estimator_
best_svc = best_clf_svc.best_estimator_
best_rf = best_clf_rf.best_estimator_
best_xgb = best_clf_xgb.best_estimator_

voting_clf_hard = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'hard') 
voting_clf_soft = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'soft') 
voting_clf_all = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('lr', best_lr)], voting = 'soft') 
voting_clf_xgb = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('xgb', best_xgb),('lr', best_lr)], voting = 'soft')

print('voting_clf_hard :',cross_val_score(voting_clf_hard,X_train,y_train,cv=5))
print('voting_clf_hard mean :',cross_val_score(voting_clf_hard,X_train,y_train,cv=5).mean())

print('voting_clf_soft :',cross_val_score(voting_clf_soft,X_train,y_train,cv=5))
print('voting_clf_soft mean :',cross_val_score(voting_clf_soft,X_train,y_train,cv=5).mean())

print('voting_clf_all :',cross_val_score(voting_clf_all,X_train,y_train,cv=5))
print('voting_clf_all mean :',cross_val_score(voting_clf_all,X_train,y_train,cv=5).mean())

print('voting_clf_xgb :',cross_val_score(voting_clf_xgb,X_train,y_train,cv=5))
print('voting_clf_xgb mean :',cross_val_score(voting_clf_xgb,X_train,y_train,cv=5).mean())

voting_clf_hard.fit(X_train, y_train)
best_rf.fit(X_train, y_train)
voting_clf_soft.fit(X_train, y_train)
voting_clf_all.fit(X_train, y_train)
voting_clf_xgb.fit(X_train, y_train)


y_hat_vc_hard = voting_clf_hard.predict(X_test).astype(int)
y_hat_rf = best_rf.predict(X_test).astype(int)
y_hat_vc_soft =  voting_clf_soft.predict(X_test).astype(int)
y_hat_vc_all = voting_clf_all.predict(X_test).astype(int)
y_hat_vc_xgb = voting_clf_xgb.predict(X_test).astype(int)


final_data = {'PassengerId': test.PassengerId, 'Survived': y_hat_rf}
submission = pd.DataFrame(data=final_data)

final_data_2 = {'PassengerId': test.PassengerId, 'Survived': y_hat_vc_hard}
submission_2 = pd.DataFrame(data=final_data_2)

final_data_3 = {'PassengerId': test.PassengerId, 'Survived': y_hat_vc_soft}
submission_3 = pd.DataFrame(data=final_data_3)

final_data_4 = {'PassengerId': test.PassengerId, 'Survived': y_hat_vc_all}
submission_4 = pd.DataFrame(data=final_data_4)

final_data_5 = {'PassengerId': test.PassengerId, 'Survived': y_hat_vc_xgb}
submission_5 = pd.DataFrame(data=final_data_5)




submission.to_csv('submission_rf.csv', index =False)
submission_2.to_csv('submission_vc_hard.csv',index=False)
submission_3.to_csv('submission_vc_soft.csv', index=False)
submission_4.to_csv('submission_vc_all.csv', index=False)
submission_5.to_csv('submission_vc_xgb2.csv', index=False)