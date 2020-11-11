import requests
from bs4 import BeautifulSoup
import pandas as pd 
import csv
import time

def ilosc_stron():
    start_url = 'https://ogloszenia.trojmiasto.pl/nieruchomosci-rynek-wtorny/mieszkanie/gdansk/ikl,101.html?strona=1'
    download_html = requests.get(start_url)

    soup = BeautifulSoup(download_html.text,features="lxml")
    with open('downloaded2.html','w',encoding='utf-8') as file:
        file.write(soup.prettify())

    page2 = soup.find('div',{'class':'form-heading'})
    page2 = page2.find('h1',{'class':'title'})
    page2 = page2.find(name='span')
    page2 = str(page2.string)
    page2 = page2[2:-1]
    page2 = int(page2)
    if page2%30==0:
        page2 = page2=page2//30
    else:
        page2 = page2//30+1                              
    
    return page2



def pobieranie_strony (page):


    start_url = page
    download_html = requests.get(start_url)

    soup = BeautifulSoup(download_html.text,features="lxml")
    with open('downloaded2.html','w',encoding='utf-8') as file:
        file.write(soup.prettify())
    
    data = soup.findAll('div',{'class':'list__item__wrap__content'})

    return data    
#______________________________________________________________________

def rok_budowy2(a,b):
    rok_budowy = ogloszenie.find(a,b)
    if rok_budowy == None:
            rok_budowy = 0
    else:
            rok_budowy = rok_budowy.find('p',{'class':'list__item__details__icons__element__desc'})
            rok_budowy = str(rok_budowy.string)
    return rok_budowy

df1 = pd.DataFrame()
df2 = pd.DataFrame()
lista = []
a = 0

for i in range(0,ilosc_stron()):    

    i = str(i)
    page3 = 'https://ogloszenia.trojmiasto.pl/nieruchomosci-rynek-wtorny/mieszkanie/gdansk/ikl,101.html?strona='
    page3 = page3+i

    strona = pobieranie_strony(page3)

    for ogloszenie in strona:

            mieszkanie = []
            
            miejsce = ogloszenie.find('p',{'class':'list__item__content__subtitle'})
            if miejsce == None:
                miejsce = 0
            else:
                miejsce = str(miejsce.string)
            mieszkanie.append(miejsce)

            link = ogloszenie.find('a',href=True)
            if link == None:
                tytul = 0
            else:
                tytul = link['title']
            link=link['href']
            mieszkanie.append(tytul)
          
            mieszkanie.append(rok_budowy2('li',{'class':'list__item__details__icons__element details--icons--element--powierzchnia'}))

            mieszkanie.append(rok_budowy2('li',{'class':'list__item__details__icons__element details--icons--element--l_pokoi'}))

            mieszkanie.append(rok_budowy2('li',{'class':'list__item__details__icons__element details--icons--element--pietro'}))

            mieszkanie.append(rok_budowy2('li',{'class':'list__item__details__icons__element details--icons--element--rok_budowy'}))

            cena = ogloszenie.find('p',{'class':'list__item__price__value'})
            cena = str(cena.string)
            mieszkanie.append(cena)
            
            df2 = pd.DataFrame ([mieszkanie],columns = ['Miejsce','Tytuł','Powierzchnia','L_pokoi','Pietro','Rok_budowy','Cena'])
            df2.set_index(['Tytuł'], inplace=True)
            df1 = pd.concat([df1,df2])
            
            
           
            
            a +=1
            if a%500 ==0:
                time.sleep(15)
                print ('1')


df1.to_csv('Portfolio/Scraping mieszkania/mieszkania pandas.csv', encoding = 'utf-8')
print(df1)