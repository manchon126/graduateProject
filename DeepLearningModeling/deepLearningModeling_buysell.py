from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer

from numpy import array
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.utils import np_utils

import matplotlib.pyplot as plt



def fromChartTime2NewsTime(chartTimeStr):
    newsTimeStr = chartTimeStr[0:4]+"."+chartTimeStr[5:7]+"."+chartTimeStr[8:10]+". "
    if int(chartTimeStr[11:13]) < 12:
        newsTimeStr += "오전 "
        if int(chartTimeStr[11:13]) == 0:
            newsTimeStr += "12:"+chartTimeStr[14:16]
        elif int(chartTimeStr[11:13]) < 10:
            newsTimeStr += chartTimeStr[12]+":"+chartTimeStr[14:16]
        else:
            newsTimeStr += chartTimeStr[11:13]+":"+chartTimeStr[14:16]
    else:
        newsTimeStr += "오후 "
        if int(chartTimeStr[11:13]) == 12:
            newsTimeStr += "12:"+chartTimeStr[14:16]
        else:
            newsTimeStr += str(int(chartTimeStr[11:13])-12)+":"+chartTimeStr[14:16]
            
           

    return newsTimeStr




dfChart = pd.read_csv("1MinuteCandle_KRW-BTC_2022.01_2021.12.csv")
dfNews = pd.read_csv("NewsSearchKeyword_비트코인_2021.12_2021.12.csv")


nOfSamples = 11



#print(dfChart[0:10])
#print("\n\n\n***\n\n\n")
#(dfNews[0:5])

datasetChart = dfChart.values
datasetNews = dfNews.values


chartIdxConverter = {}
idx = 0
for datasetChartRow in datasetChart:
    newsTimeStr = fromChartTime2NewsTime(datasetChartRow[3])
    chartIdxConverter[newsTimeStr] = idx
    idx += 1
    

    
    
newsTextSet = []
buySellIndicator=[]
highestTradePriceSampleCategorizationList = []

for datasetNewsRow in datasetNews:
    try:
        if datasetNewsRow[2] not in chartIdxConverter:
            raise Exception("시세 데이터셋에 누락된 시간")
        chartIdx = chartIdxConverter[datasetNewsRow[2]]
    except Exception as e:
        print("ERROR: ", e)
        
    #print(datasetNewsRow[2])
    
    
    newsTextSet.append(datasetNewsRow[3])
    

    
    barometer = 0
    newsPublishedTimeTradePrice = datasetChart[chartIdx][7]
    
    highestTradePrice = 0
    highestTradePriceSampleIdx = 0

    
    chartIdxPrev = chartIdxNow = chartIdx
    
    
    accumulatedTradePrice = 0
    chartIdxNow -= 1
    j = chartIdxNow
    accumulatedTradePrice += datasetChart[j][9]
    #print(datasetChart[j][3])
    #print(datasetChart[j][9])
            
    #print("sigma")
    #print(accumulatedTradePrice)      
        
    
    riseFallRatio = (datasetChart[chartIdxNow][7] - newsPublishedTimeTradePrice)/newsPublishedTimeTradePrice
    barometer += accumulatedTradePrice * riseFallRatio
    #print(riseFallRatio)
    #print(barometer)
    #print("")


    highestTradePrice = datasetChart[chartIdxNow][7]    
  
    
    chartIdxPrev = chartIdxNow


    
    #accumulatedTradePrice = 0
    for i in range(0, nOfSamples-1):
        chartIdxNow -= 2 ** i        
        for j in range(chartIdxNow, chartIdxPrev):
            accumulatedTradePrice += datasetChart[j][9]
            #print(datasetChart[j][3])
            #print(datasetChart[j][9])
            
        #print("sigma")
        #print(accumulatedTradePrice)

        
        
        riseFallRatio = (datasetChart[chartIdxNow][7] - newsPublishedTimeTradePrice)/newsPublishedTimeTradePrice
        coefficient = 2 ** (-(i+1))
        barometer += accumulatedTradePrice * riseFallRatio * coefficient
        #print(riseFallRatio)
        #print(coefficient)
        #print(barometer)
        #print("")
    

        if datasetChart[chartIdxNow][7] > highestTradePrice:
          highestTradePrice = datasetChart[chartIdxNow][7]
          highestTradePriceSampleIdx = i + 1


    
        chartIdxPrev = chartIdxNow
        
    #print(barometer)
    buySellIndicator.append(barometer)

    


    highestTradePriceSampleCategorizationList.append(highestTradePriceSampleIdx)



        
#print(newsTextSet)
#print(buySellIndicator)

maxBuySellindicatorABS = 0
for i in buySellIndicator:
    if abs(i) > maxBuySellindicatorABS:
        maxBuySellindicatorABS = abs(i)
        
        
for i in range(0, len(buySellIndicator)):
    buySellIndicator[i] /= maxBuySellindicatorABS
    
#print(buySellIndicator)

buySellSuggestion=[]

for i in range(0, len(buySellIndicator)):
    if buySellIndicator[i] >= 0.7:
        buySellSuggestion.append(0)
    elif buySellIndicator[i] >= 0.4:
        buySellSuggestion.append(1)
    elif buySellIndicator[i] >= 0.1:
        buySellSuggestion.append(2)
    elif buySellIndicator[i] >= -0.1:
        buySellSuggestion.append(3)
    elif buySellIndicator[i] >= -0.4:
        buySellSuggestion.append(4)
    elif buySellIndicator[i] >= -0.7:
        buySellSuggestion.append(5)
    else:
        buySellSuggestion.append(6)
        
#print(buySellSuggestion)

#print(len(newsTextSet))
#print(len(buySellIndicator))





token = Tokenizer()
token.fit_on_texts(newsTextSet)
#print("단어 카운트:\n", token.word_counts)

#print("\n문장 카운트: ", token.document_count)
#print("\n각 단어가 몇 개의 문장에 포함되어 있는가:\n", token.word_docs)
#print("\n각 단어에 매겨진 인덱스 값:\n", token.word_index)

x = token.texts_to_sequences(newsTextSet)
#print(x)
x_maxLen = 0
for i in x:
    if len(i) > x_maxLen:
        x_maxLen = len(i)
        
#print("x_maxLen: " , x_maxLen)
padded_x = pad_sequences(x, x_maxLen)
#print(padded_x)

word_size = len(token.word_index) + 1


seed=0
np.random.seed(seed)
tf.random.set_seed(3)

buySellSuggestion_encoded = np_utils.to_categorical(buySellSuggestion)
highestTradePriceSampleCategorizationList_encoded = np_utils.to_categorical(highestTradePriceSampleCategorizationList)

X_train0, X_test0, Y_train0, Y_test0 = train_test_split(padded_x, buySellSuggestion_encoded, test_size=0.3, random_state=seed)
#Y_train_encoded = np_utils.to_categorical(Y_train)
#Y_test_encoded = np_utils.to_categorical(Y_test)
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(padded_x, highestTradePriceSampleCategorizationList_encoded, test_size=0.3, random_state=seed)



model0 = Sequential()
model0.add(Embedding(word_size, 256, input_length=x_maxLen))
#model.add(Flatten())
#model.add(Dense(1, activation = 'sigmoid'))
model0.add(LSTM(256, activation='tanh'))
model0.add(Dense(7, activation='softmax'))

model0.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#print(X_train0.shape)
#print(X_test0.shape)
#print(Y_train0.shape)
#print(Y_test0.shape)

history = model0.fit(X_train0, Y_train0, epochs=20, batch_size=100, validation_data=(X_test0, Y_test0))


print("\n Test Accuracy: %.4f" % (model0.evaluate(X_test0, Y_test0)[1]))


y_vloss = history.history['val_loss']


y_loss = history.history['loss']


x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label="Testset_loss")
plt.plot(x_len, y_loss, marker='.', c="blue", label="trainset_loss")


plt.legend(loc="upper right")
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()




model1 = Sequential()
model1.add(Embedding(word_size, 256, input_length=x_maxLen))
#model.add(Flatten())
#model.add(Dense(1, activation = 'sigmoid'))
model1.add(LSTM(256, activation='tanh'))
model1.add(Dense(nOfSamples, activation='softmax'))

model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model1.fit(X_train1, Y_train1, epochs=20, batch_size=100, validation_data=(X_test1, Y_test1))


print("\n Test Accuracy: %.4f" % (model1.evaluate(X_test1, Y_test1)[1]))


y_vloss = history.history['val_loss']


y_loss = history.history['loss']


x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label="Testset_loss")
plt.plot(x_len, y_loss, marker='.', c="blue", label="trainset_loss")


plt.legend(loc="upper right")
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()