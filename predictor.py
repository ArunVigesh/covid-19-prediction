import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from fastai.tabular import add_datepart
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime, date, time, timezone,timedelta
import io
from sklearn.preprocessing import LabelEncoder
import json
pd.options.mode.chained_assignment = None

def getConfirmed():
    data=pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
    cnt=data[['Province/State','Country/Region','Lat','Long']]
    cnt['Key']=cnt.index
    temp=data.drop(['Province/State','Country/Region','Lat','Long'],axis=1)
    lst=[]
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            tls=[i,temp.columns[j],temp.iloc[i,j],i]
            lst.append(tls)
    df_lst=pd.DataFrame(lst)
    df_lst.columns=['Key','Date','Confirmed','loc_id']
    datas = cnt.merge(df_lst,how='inner',on=['Key'])
    datas.columns=['Province/State','Country','latitude','longitude','Key','Date','Confirmed','loc_id']
    return datas

def getDeaths():
    data=pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
    cnt=data[['Province/State','Country/Region','Lat','Long']]
    cnt['Key']=cnt.index
    temp=data.drop(['Province/State','Country/Region','Lat','Long'],axis=1)
    lst=[]
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            tls=[i,temp.columns[j],temp.iloc[i,j],i]
            lst.append(tls)
    df_lst=pd.DataFrame(lst)
    df_lst.columns=['Key','Date','Deaths','loc_id']
    datas = cnt.merge(df_lst,how='inner',on=['Key'])
    datas.columns=['Province/State','Country','latitude','longitude','Key','Date','Deaths','loc_id']
    return datas

def getRecovered():
    data=pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
    cnt=data[['Province/State','Country/Region','Lat','Long']]
    cnt['Key']=cnt.index
    temp=data.drop(['Province/State','Country/Region','Lat','Long'],axis=1)
    lst=[]
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            tls=[i,temp.columns[j],temp.iloc[i,j],i]
            lst.append(tls)
    df_lst=pd.DataFrame(lst)
    df_lst.columns=['Key','Date','Recovered','loc_id']
    datas = cnt.merge(df_lst,how='inner',on=['Key'])
    datas.columns=['Province/State','Country','latitude','longitude','Key','Date','Recovered','loc_id']
    return datas

def transform(data):
    data['Date']=pd.to_datetime(data['Date'], errors='coerce')
    data['Country']= data['Country'].apply(lambda x: 'Azerbaijan' if x==' Azerbaijan' else x)
    data['Province/State']=data['Province/State'].fillna('Unknown')
    return data

def convertCumultoLineItem(data):
    df=data.groupby(['Country','Province/State','Key','latitude','longitude','loc_id'])[['Confirmed','Deaths','Recovered']].diff()
    df = df[['Confirmed','Deaths','Recovered']]
    df.columns=['ConfirmedN','DeathsN','RecoveredN']
    data['KeyN'] = [i for i in range(data.shape[0])]
    df['KeyN'] = [i for i in range(df.shape[0])]
    data=data.merge(df,how='inner',on=['KeyN'])
    data['ConfirmedN'].fillna(data['Confirmed'],inplace=True)
    data['DeathsN'].fillna(data['Deaths'],inplace=True)
    data['RecoveredN'].fillna(data['Recovered'],inplace=True)
    data=data.drop(['Confirmed','Deaths','Recovered'],axis=1)
    data.rename(columns={
        'ConfirmedN':'Confirmed',
        'DeathsN':'Deaths',
        'RecoveredN':'Recovered'},inplace=True)
    return data

def init():
    data_c = getConfirmed()
    data_c.rename(columns={"Key":"KeyC"},inplace=True)
    data_d = getDeaths()
    data_d=data_d[['Key','Deaths']]
    data_r = getRecovered()
    data_r=data_r[['Province/State','Country','latitude','longitude','Date','Recovered']]
    data=pd.concat([data_c,data_d],axis=1)
    data = transform(data)
    data_r = transform(data_r)
    data = data.merge(data_r,on=['Province/State','Country','latitude','longitude','Date'],how='left')
    data.drop(['Key'],axis=1,inplace=True)
    data.rename(columns={"KeyC":"Key"},inplace=True)
    data = convertCumultoLineItem(data)
    data=data.drop(['Key'],axis=1)
    data=data.rename(columns={
        "KeyN":"Key"
    })
    return data

def country_data():
  df = init()
  countries = df.Country.unique()
  df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
  df.fillna(0,inplace=True)
  prediction = {}
  for country in countries:
    confirmed = []
    deaths = []
    recovered = []
    dates = []
    for i in range(len(df.Country.values)):
      if df.iloc[i,1] == country:
        dates.append(df.iloc[i,4])
        confirmed.append(df.iloc[i,7])
        deaths.append(df.iloc[i,8])
        recovered.append(df.iloc[i,9])
    data = pd.DataFrame({'Date' : dates,'Confirmed' : confirmed,'Deaths' : deaths,'Recovered' : recovered})

    covid_confirmed=data.Confirmed.astype(int)
    for i in range(1,len(covid_confirmed)):
      covid_confirmed[i] += covid_confirmed[i-1]

    covid_death=data.Deaths.astype(int)
    for i in range(1,len(covid_death)):
      covid_death[i] += covid_death[i-1]

    covid_recovered=data.Recovered.astype(int)
    for i in range(1,len(covid_recovered)):
      covid_recovered[i] += covid_recovered[i-1]

    Newly_reported=[covid_confirmed[0]]
    New_deaths=[covid_death[0]]
    New_recovered=[covid_recovered[0]]

    for i in range(1,len(covid_confirmed)):
      Newly_reported.append(covid_confirmed[i]-covid_confirmed[i-1])
      New_deaths.append(covid_death[i]-covid_death[i-1])
      New_recovered.append(covid_recovered[i]-covid_recovered[i-1])

    data['Confirmed']=covid_confirmed 
    data['Deaths']=covid_death
    data['Recovered']=covid_recovered
    data['New_Confirmed']=Newly_reported 
    data['New_Deaths']=New_deaths
    data['New_Recovered']=New_recovered
    df1_date = data[['Date','Confirmed','Deaths','Recovered']]
    df1_date['Date'] = pd.to_datetime(df1_date.Date,format='%Y-%m-%d')
    df1_date.index = df1_date['Date']
    df1_date['Date'] = pd.to_datetime(df1_date.Date,format='%Y-%m-%d')
    df1_date.index = df1_date['Date']
    data = df1_date.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df1_date)),columns=['Date', 'Confirmed'])
    for i in range(0,len(data)):
      new_data['Date'][i] = data['Date'][i]
      new_data['Confirmed'][i] = data['Confirmed'][i]
    result = adfuller(new_data.Confirmed.dropna())
    model = ARIMA(new_data.Confirmed.astype(float), order=(1,1,0))
    model_fit = model.fit(disp=0)
    prediction[country] = model_fit.forecast(1)[0][0]
  return prediction

class Predictor:
  def __init__(self):
    pass

  def train(self):
    print('Training')
    pred = country_data()
    predicted = json.dumps(pred)
    return predicted


  def predict(self):
    prediction = self.train()
    print('Forecasting')
    return prediction

