import pandas as pd 
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df1 = pd.read_csv("Portfolio/Regresja -mieszkania/mieszkania.csv",delimiter=';')

#data cleaning
#_____________________________________________

def remove_czm_outliners(df): #usunięcie rekordów różniących się o więcej niz odchylenie standardowe od średniej 
    df_out=pd.DataFrame()
    for key, subdf in df.groupby('miejsce'):
        m = np.mean(subdf.cena_za_metr)
        st = np.std(subdf.cena_za_metr)
        reduced_df = subdf[(subdf.cena_za_metr>(m-st))&(subdf.cena_za_metr<=(m+st))]
        df_out = pd.concat([df_out, reduced_df],ignore_index=True)
    return df_out


df1 = df1.drop(['tytul','rok_budowy','Ulica'],axis='columns') #wybranie odpowiednich kolumn
df1 = df1.dropna() #usunięcie rekordów z brakami 
df1 = df1[~(df1.powierzchnia/df1.l_pokoi<5)]  #usunięcie rekordu z zbyt małą ilością metrów na pokój
df1 ['cena_za_metr'] = (df1['cena']/df1['powierzchnia']).round(0)
df1 = remove_czm_outliners(df1) # usunięcie rekordów mających wartości ceny za metr różniące się o więcej niż jedno odchylenie od średniej


dummies = pd.get_dummies(df1.miejsce)
df1 = pd.concat([df1,dummies],axis='columns')
df1 = df1.drop('miejsce',axis='columns')

print(df1)


# X = df1.drop(['cena','cena_za_metr'], axis = 'columns')
# y = df1.cena
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)



# def predict_price(miejsce,powierzchnia,l_pokoi,pietro):

#     lr_clf = LinearRegression(normalize= False)
#     lr_clf.fit(X_train,y_train)
#     lr_clf.score(X_test,y_test)
#     loc_index = np.where(X.columns==miejsce)[0][0]
#     x = np.zeros(len(X.columns))
#     x[0] = powierzchnia
#     x[1] = l_pokoi
#     x[2] = pietro
#     if loc_index>=0:
#         x[loc_index] = 1
#     return lr_clf.predict([x])[0] #metoda predict



# print(predict_price('Gdańsk Przymorze',58.5,3,2))
# print(predict_price('Gdańsk Oliwa',30,2,1))