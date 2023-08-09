#!pip uninstall JWT
#!pip uninstall PyJWT
!pip install PyJWT

import os
import jwt
import uuid
import hashlib
from urllib.parse import urlencode

import requests

import time



from selenium import webdriver
#import time
from bs4 import BeautifulSoup
import datetime
from datetime import timedelta
import pandas as pd


from tensorflow.keras.models import load_model


#import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer

from numpy import array
from tensorflow.keras.preprocessing.sequence import pad_sequences



def getCoinBalance(access_key, secret_key, server_url, currencyCode):
    access_key = access_key
    secret_key = secret_key
    server_url = server_url
    
    coinBalance = 0.0
    
    payload = {
        'access_key': access_key,
        'nonce': str(uuid.uuid4()),
    }

    jwt_token = jwt.encode(payload, secret_key)
    authorize_token = 'Bearer {}'.format(jwt_token)
    headers = {"Authorization": authorize_token}

    res = requests.get(server_url + "/v1/accounts", headers=headers)

    res_json = res.json()
    
    for i in range(0, len(res_json)):
        if res_json[i]['currency'] == currencyCode:
            coinBalance = float(res_json[i]['balance'])
  
    return coinBalance



def addToOrdersList(ordersList, newOrder_time, newOrder_buySell, newOrder_coinAmount):
    ordersList.append([0, 0, 0])
    idx = 0
    
    for idx in range(len(ordersList)-1, 0, -1):
        if ordersList[idx-1][0] > newOrder_time:
            ordersList[idx] = ordersList[idx-1]
            if idx == 1:
                idx = 0
        else:
            break;
            
        
    ordersList[idx] = [newOrder_time, newOrder_buySell, newOrder_coinAmount]
    
    
def getFirstOfOrdersList(ordersList):
    if len(ordersList) == 0:
        return [-1, -1, -1]
    
    return ordersList[0]


def deleteFirstOfOrdersList(ordersList):
    if len(ordersList) == 0:
        return [-1, -1, -1]
    
    firstElementOfOrdersList = ordersList[0]
    
    for i in range(1, len(ordersList)):
        ordersList[i-1] = ordersList[i]
        
    ordersList.pop()
    
    return firstElementOfOrdersList
    
    
def makeBuy_MarketPriceOrder(access_key, secret_key, server_url, market, amount, currencyCode):    
    access_key = access_key
    secret_key = secret_key
    server_url = server_url

    
    KRWBalanceBefore = getCoinBalance(access_key, secret_key, server_url, "KRW")
    coinBalanceBefore = getCoinBalance(access_key, secret_key, server_url, currencyCode)
    
    
    query = {
        'market': market,
        'side': 'bid',
        'price': amount,
        'ord_type': 'price',
    }
    query_string = urlencode(query).encode()

    m = hashlib.sha512()
    m.update(query_string)
    query_hash = m.hexdigest()

    payload = {
        'access_key': access_key,
        'nonce': str(uuid.uuid4()),
        'query_hash': query_hash,
        'query_hash_alg': 'SHA512',
    }

    jwt_token = jwt.encode(payload, secret_key)
    authorize_token = 'Bearer {}'.format(jwt_token)
    headers = {"Authorization": authorize_token}

    res = requests.post(server_url + "/v1/orders", params=query, headers=headers)


    KRWBalanceAfter = getCoinBalance(access_key, secret_key, server_url, "KRW")   
    coinBalanceAfter = getCoinBalance(access_key, secret_key, server_url, currencyCode) 
    KRWBalanceDiff = KRWBalanceAfter - KRWBalanceBefore
    coinBalanceDiff = coinBalanceAfter - coinBalanceBefore
    
    print("Buy ",market," , KRW: ",KRWBalanceDiff," , Coin: ",coinBalanceDiff," @market price")
    
    
    return [KRWBalanceDiff, coinBalanceDiff]

    
def makeSell_MarketPriceOrder(access_key, secret_key, server_url, market, volume, currencyCode):    
    access_key = access_key
    secret_key = secret_key
    server_url = server_url
    

    KRWBalanceBefore = getCoinBalance(access_key, secret_key, server_url, "KRW")
    coinBalanceBefore = getCoinBalance(access_key, secret_key, server_url, currencyCode)
    
    
    query = {
        'market': market,
        'side': 'ask',
        'volume': volume,
        'ord_type': 'market',
    }
    query_string = urlencode(query).encode()

    m = hashlib.sha512()
    m.update(query_string)
    query_hash = m.hexdigest()

    payload = {
        'access_key': access_key,
        'nonce': str(uuid.uuid4()),
        'query_hash': query_hash,
        'query_hash_alg': 'SHA512',
    }

    jwt_token = jwt.encode(payload, secret_key)
    authorize_token = 'Bearer {}'.format(jwt_token)
    headers = {"Authorization": authorize_token}

    res = requests.post(server_url + "/v1/orders", params=query, headers=headers)


    
    KRWBalanceAfter = getCoinBalance(access_key, secret_key, server_url, "KRW")   
    coinBalanceAfter = getCoinBalance(access_key, secret_key, server_url, currencyCode) 
    KRWBalanceDiff = KRWBalanceAfter - KRWBalanceBefore
    coinBalanceDiff = coinBalanceAfter - coinBalanceBefore
    
    print("Sell ",market," , KRW: ",KRWBalanceDiff," , Coin: ",coinBalanceDiff," @market price")
    
    return [KRWBalanceDiff, coinBalanceDiff]







os.environ['UPBIT_OPEN_API_ACCESS_KEY'] = "UC4uPBtI1nIGGS8ZYIXjPagQBiOAxKx4LJMkWgx9"
access_key = os.environ['UPBIT_OPEN_API_ACCESS_KEY']
os.environ['UPBIT_OPEN_API_SECRET_KEY'] = "wi1dRxWpYXGTmYQBOjuZr5SOHukHSxL2TZFYU2Rr"
secret_key = os.environ['UPBIT_OPEN_API_SECRET_KEY']
os.environ['UPBIT_OPEN_API_SERVER_URL'] = "https://api.upbit.com"
server_url = os.environ['UPBIT_OPEN_API_SERVER_URL']


KRWBalance = 600000.0
BTCBalance = 0.0

ordersList = []

try:
    pd_ordersList = pd.read_csv('ordersList.csv')
    print("pd_ordersList: ")
    print(pd_ordersList)
    pd_ordersList_values = pd_ordersList.values
    
    for povRow in pd_ordersList_values:
        ordersList.append([povRow[1], povRow[2], povRow[3]])
        
except:
    print("ERROR: .csv file of orders list is empty")


print("ordersList: ")
print(ordersList)



try:
    pd_krwCoinBalance = pd.read_csv('krwCoinBalance.csv')
    print("pd_krwCoinBalance: ")
    print(pd_krwCoinBalance)
    pd_krwCoinBalance = pd_krwCoinBalance.values
    
    KRWBalance = pd_krwCoinBalance[0][1]
    BTCBalance = pd_krwCoinBalance[1][1]
    
       
except:
    print("ERROR: .csv file of balances is empty")
    

print("KRWBalance: ", KRWBalance)
print("BTCBalance: ", BTCBalance)






chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
#driver = webdriver.Chrome('chromedriver', chrome_options=chrome_options)
#driver = webdriver.Chrome()
driver = webdriver.Chrome('C:\\Users\\AoiMirai\\Downloads\\chromedriver_v101.exe', chrome_options=chrome_options)
#url = "https://kr.investing.com/news/cryptocurrency-news/article-740246"


searchKeyword = "비트코인"
               
    
newsSearchKeyword = "비트코인"

newsStartYear = "2021"
newsStartMonth = "01"
newsEndYear = "2021"
newsEndMonth = "03"


latestNewsLink = ""


        
modelDir = "./models/"
model0 = load_model(modelDir+"비트코인_2021.01_2021.03_model0.hdf5")
model1 = load_model(modelDir+"비트코인_2021.01_2021.03_model1_14.hdf5")
model2_4 = load_model(modelDir+"비트코인_2021.01_2021.03_model2_4.hdf5")
model2_10 = load_model(modelDir+"비트코인_2021.01_2021.03_model2_10.hdf5")
model2_15 = load_model(modelDir+"비트코인_2021.01_2021.03_model2_15.hdf5")
model3_4 = load_model(modelDir+"비트코인_2021.01_2021.03_model3_4.hdf5")
model3_10 = load_model(modelDir+"비트코인_2021.01_2021.03_model3_10.hdf5")
model3_15 = load_model(modelDir+"비트코인_2021.01_2021.03_model3_15.hdf5")








while True:
    print("Time: ",time.time())
    nowTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(nowTime)
    print("KRWBalance: ",KRWBalance)        
    print("BTCBalance: ",BTCBalance)    
    
    
    while True:
        nearestOrder = getFirstOfOrdersList(ordersList)
        print("nearestOrder: ")
        print(nearestOrder)
        nearestOrderTimeOfOrdersList = nearestOrder[0]
        if nearestOrderTimeOfOrdersList != -1 and nearestOrderTimeOfOrdersList < time.time():
            if nearestOrder[1] == "SELL":
                krwCoinDiff = makeSell_MarketPriceOrder(access_key, secret_key, server_url, "KRW-BTC", nearestOrder[2], "BTC")
                KRWBalance += krwCoinDiff[0]
                BTCBalance += krwCoinDiff[1]
                deleteFirstOfOrdersList(ordersList)
            elif  nearestOrder[1] == "BUY":
                krwCoinDiff = makeBuy_MarketPriceOrder(access_key, secret_key, server_url, "KRW-BTC", nearestOrder[2], "BTC")                
                KRWBalance += krwCoinDiff[0]
                BTCBalance += krwCoinDiff[1]
                deleteFirstOfOrdersList(ordersList)
                
            
            np_ordersList = np.asarray(ordersList)
            pd.DataFrame(np_ordersList).to_csv('ordersList.csv')


            np_krwCoin = np.asarray([KRWBalance, BTCBalance])
            pd.DataFrame(np_krwCoin).to_csv('krwCoinBalance.csv')    
                
    
        else:
            break   
    
    
    
    
    
    
        
    url = "https://search.naver.com/search.naver?where=news&query="+searchKeyword+"&sm=tab_opt&sort=1&photo=0&field=0&pd=0&ds=&de=&docid=&related=0&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so%3Add%2Cp%3Aall&is_sug_officeid=0"
    print(url)
    driver.get(url)
    

    

    #for i in range(0, 5):
    html = driver.page_source

    soup = BeautifulSoup(html, 'html.parser')

    try:
        newsList = soup.select_one(".list_news")
        newsLinkList = newsList.select("li > div > div > div > div > a")
        #print(newsList)
    except:        
        print("News does not exist")
        time.sleep(60)
        continue;
        

    naverNewsLink = ""
    idxx = 0
    for newslink in newsLinkList:
        link = newslink['href']
        if link[:23] == "https://news.naver.com/":
            naverNewsLink = link
            break;


    print("naverNewsLink: ", end="")
    if naverNewsLink == "":
        print("Naver news does not exist")
        time.sleep(60)
        continue;
    elif naverNewsLink == latestNewsLink:
        print("Latest news is not changed")
        time.sleep(60)
        continue;
    else:
        print(naverNewsLink)        
        latestNewsLink = naverNewsLink
    
        
    

    driver.get(naverNewsLink)
    

    html = driver.page_source


    soup = BeautifulSoup(html, 'html.parser')

    try:
        newsTitle = soup.select_one(".media_end_head_headline")
        newsTitle = newsTitle.text

        newsTime = soup.select_one(".media_end_head_info_datestamp_time")
        newsTime = newsTime.text

        newsContent = soup.select_one("#newsct_article")
        newsContent = newsContent.text
    except:
        try:
            newsTitle = soup.select_one(".end_tit")
            newsTitle = newsTitle.text

            newsTime = soup.select_one(".author>em")
            newsTime = newsTime.text

            newsContent = soup.select_one("#articeBody")
            newsContent = newsContent.text
        except:
            newsTitle = soup.select_one(".title")
            newsTitle = newsTitle.text

            newsTime = soup.select_one(".info>span")
            newsTime = newsTime.text

            newsContent = soup.select_one(".news_end font1 size3")
            newsContent = newsContent.text
                

    print([newsTitle, newsTime, newsContent])

        
    dfNews = pd.read_csv("NewsSearchKeyword_"+newsSearchKeyword+"_"+newsStartYear+"."+newsStartMonth+"_"+newsEndYear+"."+newsEndMonth+".csv")


    datasetNews = dfNews.values
    
        
    newsTextSet = []
    for datasetNewsRow in datasetNews:
        newsTextSet.append(datasetNewsRow[3])
        
    token0 = Tokenizer()
    token0.fit_on_texts(newsTextSet)
    
    lenOfToken0_word_index = len(token0.word_index)
    

    x = token0.texts_to_sequences(newsTextSet)
    
    x_maxLen = 0
    for i in range(0, len(x)-1):
        if len(x[i]) > x_maxLen:
            x_maxLen = len(x[i])

    padded_x = pad_sequences(x, x_maxLen)
    x_idx = 0
    for x_a in padded_x:
        x_idx += 1

    word_size = len(token0.word_index) + 1  
    
    
    
    
    
    newsTextSet.append(newsContent)    
        
    token1 = Tokenizer()
    token1.fit_on_texts(newsTextSet)

    x1 = token1.texts_to_sequences(newsTextSet)
    
    recentNews_texts_to_sequences = x1[-1]
    temp_recentNews_texts_to_sequences = []
    for i in recentNews_texts_to_sequences:
        if i < lenOfToken0_word_index: 
            temp_recentNews_texts_to_sequences.append(i)
    
    x.append(temp_recentNews_texts_to_sequences)
            
    
    
    padded_x = pad_sequences(x, x_maxLen)

    word_size = len(token1.word_index) + 1    
    
        
    prediction0 = model0.predict(padded_x)    
    prediction0 = prediction0[-1]
    
    prediction1 = model1.predict(padded_x)
    prediction1 = prediction1[-1]
    prediction1[0] = -1 #배열의 맨 앞은 관망일 때 이므로 매도매수시점 예측이 아님 -> sentinel
    prediction1[-1] = -1 #전부 배열의 맨 끝만 나와서 배제함
    lenOfSellBuyTimeBelt = len(prediction1)
    
    prediction2_4 = model2_4.predict(padded_x)
    prediction2_4 = prediction2_4[-1]
    prediction2_10 = model2_10.predict(padded_x)
    prediction2_10 = prediction2_10[-1]
    prediction2_15 = model2_15.predict(padded_x)
    prediction2_15 = prediction2_15[-1]
    prediction3_4 = model3_4.predict(padded_x)
    prediction3_4 = prediction3_4[-1]
    prediction3_10 = model3_10.predict(padded_x)
    prediction3_10 = prediction3_10[-1]
    prediction3_15 = model3_15.predict(padded_x)
    prediction3_15 = prediction3_15[-1]
        
    
    predictionList = [prediction0, prediction1, prediction2_4, prediction2_10, prediction2_15, prediction3_4, prediction3_10, prediction3_15]
    #print(predictionList)
    
    for i in range(0, len(predictionList)):
        maxValue = 0.0
        maxValueIdx = 0
        for j in range(0, len(predictionList[i])):
            if predictionList[i][j] > maxValue:
                maxValue = predictionList[i][j]
                maxValueIdx = j
        predictionList[i] = maxValueIdx
    
    
    print(predictionList)
    
    
    
    buySellBalance = 0
    buySellVolume = 0
    
    
    buySellLevelAvrg = (predictionList[0]+predictionList[2]+predictionList[3]+predictionList[4]+predictionList[5]+predictionList[6]+predictionList[7])/7
    print(buySellLevelAvrg," -> " , round(buySellLevelAvrg))
    buySellLevelAvrg = round(buySellLevelAvrg)
    
    
    if buySellLevelAvrg == 0:
        print("강한 매수")
        buySellBalance = KRWBalance * 0.5
    elif buySellLevelAvrg == 1:
        print("매수")
        buySellBalance = KRWBalance * 0.3
    elif buySellLevelAvrg == 2:
        print("약한 매수")
        buySellBalance = KRWBalance * 0.1
    elif buySellLevelAvrg == 3:
        print("관망")
    elif buySellLevelAvrg == 4:
        print("약한 매도")
        buySellVolume = BTCBalance * 0.1
    elif buySellLevelAvrg == 5:
        print("매도")
        buySellVolume = BTCBalance * 0.3
    else:
        print("강한 매도")
        buySellVolume = BTCBalance * 0.5
        
    
     
    sellBuyTimeBelt = predictionList[1]         
    print("Sell Buy Time Belt #", sellBuyTimeBelt)  
    if buySellLevelAvrg == 3:
        print()
    elif sellBuyTimeBelt == lenOfSellBuyTimeBelt-1:
        print(">=",2**sellBuyTimeBelt,"분 후")
    else:  
        print(2**sellBuyTimeBelt," ~ ",2**(sellBuyTimeBelt+1)-1," 분 후")
    
    
    sellBuyTime = 2**sellBuyTimeBelt
            
    
    if buySellLevelAvrg <= 2:
        krwCoinDiff = makeBuy_MarketPriceOrder(access_key, secret_key, server_url, "KRW-BTC", buySellBalance, 'BTC')
        KRWBalance += krwCoinDiff[0]
        BTCBalance += krwCoinDiff[1]
        orderAmount = krwCoinDiff[1]
        addToOrdersList(ordersList, time.time()+60*sellBuyTime, "SELL", orderAmount)
    elif buySellLevelAvrg >= 4:
        krwCoinDiff = makeSell_MarketPriceOrder(access_key, secret_key, server_url,"KRW-BTC", buySellVolume, "BTC")
        KRWBalance += krwCoinDiff[0]
        BTCBalance += krwCoinDiff[1]
        orderBalance = krwCoinDiff[0]
        addToOrdersList(ordersList, time.time()+60*sellBuyTime, "BUY", orderBalance)
   
    
    print("ordersList:")
    print(ordersList)
    
    np_ordersList = np.asarray(ordersList)
    pd.DataFrame(np_ordersList).to_csv('ordersList.csv')
    
    
    np_krwCoin = np.asarray([KRWBalance, BTCBalance])
    pd.DataFrame(np_krwCoin).to_csv('krwCoinBalance.csv')

        
    print("...")
        
        
    time.sleep(60)
