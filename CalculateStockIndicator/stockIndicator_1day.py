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



def from_1MinuteChartTimeToNewsTime(_1MinuteChartTimeStr):
    newsTimeStr = _1MinuteChartTimeStr[0:4]+"."+_1MinuteChartTimeStr[5:7]+"."+_1MinuteChartTimeStr[8:10]+". "
    if int(_1MinuteChartTimeStr[11:13]) < 12:
        newsTimeStr += "오전 "
        if int(_1MinuteChartTimeStr[11:13]) == 0:
            newsTimeStr += "12:"+_1MinuteChartTimeStr[14:16]
        elif int(_1MinuteChartTimeStr[11:13]) < 10:
            newsTimeStr += _1MinuteChartTimeStr[12]+":"+_1MinuteChartTimeStr[14:16]
        else:
            newsTimeStr += _1MinuteChartTimeStr[11:13]+":"+_1MinuteChartTimeStr[14:16]
    else:
        newsTimeStr += "오후 "
        if int(_1MinuteChartTimeStr[11:13]) == 12:
            newsTimeStr += "12:"+_1MinuteChartTimeStr[14:16]
        else:
            newsTimeStr += str(int(_1MinuteChartTimeStr[11:13])-12)+":"+_1MinuteChartTimeStr[14:16]
            
           

    return newsTimeStr


def fromDayChartDateToNewsDate(DayChartDateStr):
    newsDateStr = DayChartDateStr[0:4]+"."+DayChartDateStr[5:7]+"."+DayChartDateStr[8:10]
    
    return newsDateStr





df_1MinuteChart = pd.read_csv("1MinuteCandle_KRW-BTC_2022.01_2021.12.csv")
dfNews = pd.read_csv("NewsSearchKeyword_비트코인_2021.12_2021.12.csv")
dfDayChart = pd.read_csv("DayCandle_KRW-BTC_2022.01_2021.10.csv")


nOfSamples = 11



#print(df_1MinuteChart[0:10])
#print("\n\n\n***\n\n\n")
#(dfNews[0:5])

dataset_1MinuteChart = df_1MinuteChart.values
datasetNews = dfNews.values
datasetDayChart = dfDayChart.values


_1MinuteChartIdxConverter = {}
idx = 0
for dataset_1MinuteChartRow in dataset_1MinuteChart:
    newsTimeStr = from_1MinuteChartTimeToNewsTime(dataset_1MinuteChartRow[3])
    _1MinuteChartIdxConverter[newsTimeStr] = idx
    idx += 1
    

DayChartIdxConverter = {}
idx = 0 
for i in range(len(datasetDayChart)-1, -1, -1):
    #UTC 00H / KST 09H 기준
    newsDateStr = fromDayChartDateToNewsDate(datasetDayChart[i][3])
    #print(newsDateStr)
    DayChartIdxConverter[newsDateStr] = idx
    idx += 1
    
newsTextSet = []
buySellIndicator=[]
highestTradePriceSampleCategorizationList = []



#스토캐스틱 보조지표(패스트 스토캐스틱 %K, %D, 슬로우 스토캐스틱 %K, %D)
#parameter: (15-5-3)
# %K = 스토캐스틱N = (현재 가격 - N_0일중 최저가)/(N일중 최고가 - N_0일중 최저가)
#현재가격은 UTC 00H / KST 09H 기준 종가
# %D = 스토캐스틱N의 이동평균선(N_1일)
#상기 %K, %D를 아래의 슬로우 스토캐스틱과 대비해서 패스트 스토캐스틱 %K, %D라 함
#슬로우 스토캐스틱 그래프의 %K와 %D = 본래의 %K와 %D로부터 얻어진 t일 이동평균선(N_2일)  

#stochasticList = [패스트 %K, 패스트 %D, 슬로우 %K, 슬로우 %D]
stochasticParameter = [15, 5, 3]
stochasticList = []
stochasticIdx = DayChartIdxConverter["2021.11.23"]
#print(stochasticIdx)
#print(len(datasetDayChart))
coinPriceList_Stochastic = []
for i in range(stochasticIdx, len(datasetDayChart)):
    # 패스트 스토캐스틱 %K 
    currentPrice = datasetDayChart[i][7]
    NDaysLowestPrice = currentPrice
    NDaysHighestPrice = currentPrice    
    for j in range(1, stochasticParameter[0]+1):
        jDaysAgoLowPrice = datasetDayChart[i-j][6]
        #print(jDaysAgoLowPrice, end="\t")
        if jDaysAgoLowPrice < NDaysLowestPrice:
            NDaysLowestPrice = jDaysAgoLowPrice
        jDaysAgoHighPrice = datasetDayChart[i-j][5]
        #print(jDaysAgoHighPrice)
        if jDaysAgoHighPrice > NDaysHighestPrice:
            NDaysHighestPrice = jDaysAgoHighPrice    
    
    fastStochasticPercentK = (currentPrice-NDaysLowestPrice)/(NDaysHighestPrice-NDaysLowestPrice)
    #print(currentPrice,end="\t")
    #print(NDaysLowestPrice,end="\t")
    #print(NDaysHighestPrice,end="\t")
    #print(NDaysLowestPrice)
    #print(fastStochasticPercentK)
    
    stochasticList.append([fastStochasticPercentK, -1, -1, -1])
    coinPriceList_Stochastic.append(currentPrice)
    
for i in range(stochasticParameter[1]-1, len(stochasticList)):
    # 패스트 스토캐스틱 %D
    movingAverage = stochasticList[i][0]
    for j in range(i-(stochasticParameter[1]-1), i):
         movingAverage += stochasticList[j][0]
    stochasticList[i][1] = movingAverage/stochasticParameter[1]

for i in range(stochasticParameter[2]-1, len(stochasticList)):
    #슬로우 스토캐스틱 %K   
    movingAverage = stochasticList[i][0]
    for j in range(i-(stochasticParameter[2]-1), i):
         movingAverage += stochasticList[j][0]
    stochasticList[i][2] = movingAverage/stochasticParameter[2]
    
for i in range(stochasticParameter[1]+stochasticParameter[2]-2, len(stochasticList)):
     #슬로우 스토캐스틱 %D   
    movingAverage = stochasticList[i][1]
    for j in range(i-(stochasticParameter[2]-1), i):
         movingAverage += stochasticList[j][1]
    stochasticList[i][3] = movingAverage/stochasticParameter[2]
    
    

fastStochasticPercentKList = []
for i in range (0, len(stochasticList)):
    fastStochasticPercentKList.append(stochasticList[i][0])
fastStochasticPercentDList = []
for i in range (0, len(stochasticList)):
    fastStochasticPercentDList.append(stochasticList[i][1])
slowStochasticPercentKList = []
for i in range (0, len(stochasticList)):
    slowStochasticPercentKList.append(stochasticList[i][2])
slowStochasticPercentDList = []
for i in range (0, len(stochasticList)):
    slowStochasticPercentDList.append(stochasticList[i][3])
    
print(len(stochasticList))    




x_len = np.arange(len(stochasticList))
plt.plot(x_len, coinPriceList_Stochastic, marker='.', c="red", label="coinPrice_Stochastic")


plt.legend(loc="upper right")
plt.grid()
plt.xlabel('days')
plt.ylabel('value')
plt.show()


x_len = np.arange(len(stochasticList))
plt.plot(x_len, fastStochasticPercentKList, marker='.', c="red", label="fastStochastic%KList")
plt.plot(x_len, fastStochasticPercentDList, marker='.', c="blue", label="fastStochastic%DList")


plt.legend(loc="upper right")
plt.grid()
plt.xlabel('days')
plt.ylabel('value')
plt.show()
    
    
x_len = np.arange(len(stochasticList))
plt.plot(x_len, slowStochasticPercentKList, marker='.', c="red", label="slowStochastic%KList")
plt.plot(x_len, slowStochasticPercentDList, marker='.', c="blue", label="slowStochastic%DList")


plt.legend(loc="upper right")
plt.grid()
plt.xlabel('days')
plt.ylabel('value')
plt.show()








#MACD 보조지표
#parameter: (12, 26, 9)
#지수이동평균(EMA) = 금일 종가 * 승수 + 작일 EMA * 승수
#EMA 승수 : 2/(1+n)
#MACD = 단기 지수이동평균(n_0일) - 장기 지수이동평균(n_1일)
#signal : MACD의 n_2일 지수이동평균

#MACDList = [단기 지수이동평균, 장기 지수이동평균, MACD, signal]
MACDParameter = [12, 26, 9]
MACDList = []
MACDIdx = DayChartIdxConverter["2021.11.22"]
initialShortPeriodEMA = 0
initialLongPeriodEMA = 0
#print(stochasticIdx)
#print(len(datasetDayChart))
coinPriceList_MACD = []
#for i in range(MACDIdx-(MACDParameter[0]-1) , MACDIdx):
for i in range(MACDIdx-(MACDParameter[0]-1) , MACDIdx+1):
    initialShortPeriodEMA += datasetDayChart[i][7]
initialShortPeriodEMA /= MACDParameter[0]
#for i in range(MACDIdx-(MACDParameter[1]-1) , MACDIdx):
for i in range(MACDIdx-(MACDParameter[1]-1) , MACDIdx+1):
    initialLongPeriodEMA += datasetDayChart[i][7]
initialLongPeriodEMA /= MACDParameter[1]
MACDList.append([initialShortPeriodEMA, initialLongPeriodEMA, initialShortPeriodEMA-initialLongPeriodEMA, -1])
coinPriceList_MACD.append(datasetDayChart[MACDIdx][7])

for i in range(MACDIdx+1, len(datasetDayChart)):
    # 단기 지수이동평균, 장기 지수이동평균, MACD
    shortPeriodEP = 2/(MACDParameter[0]+1)
    longPeriodEP = 2/(MACDParameter[1]+1)   
    currentPrice = datasetDayChart[i][7]
    shortPeriodEMA = currentPrice * shortPeriodEP + MACDList[-1][0] * (1-shortPeriodEP)
    longPeriodEMA = currentPrice * longPeriodEP + MACDList[-1][1] * (1-longPeriodEP)
    MACD = shortPeriodEMA - longPeriodEMA
    MACDList.append([shortPeriodEMA, longPeriodEMA, MACD, -1])
    
    coinPriceList_MACD.append(currentPrice)
    
    
for i in range(MACDParameter[2]-1, len(MACDList)):
    #signal
    
    movingAverage = MACDList[i][2]
    for j in range(i-(MACDParameter[1]-1), i):
         movingAverage += MACDList[j][0]
    MACDList[i][3] = movingAverage/MACDParameter[2]
    

    
    

MACDshorttermEMAList = []
for i in range (0, len(MACDList)):
    MACDshorttermEMAList.append(MACDList[i][0])
MACDlongtermEMAList = []
for i in range (0, len(MACDList)):
    MACDlongtermEMAList.append(MACDList[i][1])
MACDMACDList = []
for i in range (0, len(MACDList)):
    MACDMACDList.append(MACDList[i][2])
MACDSignalList = []
for i in range (0, len(MACDList)):
    MACDSignalList.append(MACDList[i][3])
    
print(len(MACDList))    




x_len = np.arange(len(MACDList))
plt.plot(x_len, coinPriceList_MACD, marker='.', c="red", label="coinPrice_MACD")


plt.legend(loc="upper right")
plt.grid()
plt.xlabel('days')
plt.ylabel('value')
plt.show()


x_len = np.arange(len(MACDList))
plt.plot(x_len, MACDshorttermEMAList, marker='.', c="red", label="MACDshorttermEMAList")
plt.plot(x_len, MACDlongtermEMAList, marker='.', c="blue", label="MACDlongtermEMAList")
plt.plot(x_len, MACDMACDList, marker='o', c='green', label="MACDMACDList")


plt.legend(loc="upper right")
plt.grid()
plt.xlabel('days')
plt.ylabel('value')
plt.show()
    
    
x_len = np.arange(len(MACDList))
plt.plot(x_len, MACDSignalList, marker='.', c="blue", label="MACDSignalList")


plt.legend(loc="upper right")
plt.grid()
plt.xlabel('days')
plt.ylabel('value')
plt.show()




#RSI 보조지표
#parameter: 14
#가격이 전일 가격보다 상승한 날의 상승분은 U(up) 값, 하락한 날의 하락분은 D(down) 값
#U값과 D값의 평균값은 각각 AU(average ups)와 AD(average downs) 값
#AU를 AD값으로 나눈 것을 RS(relative strength) 값 
#RSI = RS / (1 + RS)

#RSIList = [RSI]
RSIParameter = [14]
RSIList = []
RSIIdx =  DayChartIdxConverter["2021.12.01"]
diffList = [0]
#print(stochasticIdx)
#print(len(datasetDayChart))

for i in range(1, len(datasetDayChart)):
    diffList.append(datasetDayChart[i][7]-datasetDayChart[i-1][7])   
    
      
sumOfU = 0
sumOfD = 0

coinPriceList_RSI = []
for i in range(RSIIdx-(RSIParameter[0]), RSIIdx):
    #UList, DList의 initialization
    if diffList[i] >= 0:
        sumOfU += diffList[i]
    else:
        #절대값(ABS)을 취함
        sumOfD -= diffList[i]

    

for i in range(RSIIdx, len(datasetDayChart)):
    diffOfNDaysAgo = diffList[i-RSIParameter[0]]
    diffOfToday = diffList[i]
    if diffOfNDaysAgo >= 0:
        sumOfU -= diffOfNDaysAgo
    else:
        sumOfD += diffOfNDaysAgo
    if diffOfToday >= 0:
        sumOfU += diffOfToday
    else:
        sumOfD -= diffOfToday
        
    AU = sumOfU / RSIParameter[0]
    AD = sumOfD / RSIParameter[0]
    
    if AD == 0:
        RSI = 1
    else:
        RS = AU / AD
        RSI = RS / (1 + RS)

        
    RSIList.append([RSI])
    coinPriceList_RSI.append(datasetDayChart[i][7])
    
    
    
print(len(RSIList))    


RSIRSIList = []
for i in range (0, len(RSIList)):
    RSIRSIList.append(RSIList[i][0])


x_len = np.arange(len(RSIList))
plt.plot(x_len, coinPriceList_RSI, marker='.', c="red", label="coinPriceList_RSI")


plt.legend(loc="upper right")
plt.grid()
plt.xlabel('days')
plt.ylabel('value')
plt.show()


x_len = np.arange(len(RSIList))
plt.plot(x_len, RSIRSIList, marker='.', c="red", label="RSIList")


plt.legend(loc="upper right")
plt.grid()
plt.xlabel('days')
plt.ylabel('value')
plt.show()
