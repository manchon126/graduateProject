from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM, Dropout, Conv1D, MaxPooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping


import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer

from numpy import array
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.utils import np_utils

import matplotlib.pyplot as plt

import os





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


def fromDayChartDateToNewsDate(DayChartDateStr):
    newsDateStr = DayChartDateStr[0:4]+"."+DayChartDateStr[5:7]+"."+DayChartDateStr[8:10]
    
    return newsDateStr


def fromMarketCapitalizationDateToNewsDate(MarketCapitalizationDateStr):
    monthCode = {"Jan":"01", "Feb":"02", "Mar":"03", "Apr":"04", "May":"05", "Jun":"06",
                 "Jul":"07", "Aug":"08", "Sep":"09", "Oct":"10", "Nov":"11", "Dec":"12"}
       
    newsDateStr = MarketCapitalizationDateStr[8:]+"."+monthCode[MarketCapitalizationDateStr[:3]]+"."+MarketCapitalizationDateStr[4:6]
    
    return newsDateStr




def getNextMinuteOfNewsTime(newsTimeStr):
    year = int(newsTimeStr[:4])
    month = int(newsTimeStr[5:7])
    day = int(newsTimeStr[8:10])
    ampm = newsTimeStr[12:14]
    if newsTimeStr[16] == ":":
        hour = int(newsTimeStr[15])
        minute = int(newsTimeStr[17:])
    else:
        hour = int(newsTimeStr[15:17])
        minute = int(newsTimeStr[18:])
            
    if int(minute) <= 58:
        minute = int(minute)+1
    else: #59분
        minute = 0
        if int(hour) <= 10:
            hour = int(hour)+1
        else:
            if ampm == "오전" and hour == 12:
                hour = 1
            elif ampm == "오전" and hour == 11:
                ampm = "오후"
                hour = 12
            elif ampm == "오후" and hour == 12:
                hour =1
            else: #오후11시
                ampm = "오전"
                hour = 12
                nOfDays = howmanyDays(year, month)
                if int(day) < nOfDays:
                    day = int(day)+1
                else:
                    day = 1
                    if month != 12:
                        month = int(month)+1
                    else:
                        month = 1
                        year = int(year)+1
                        
    year = str(year)
    if month < 10:
        month = "0"+str(month)
    else:
        month = str(month)
    if day < 10:
        day = "0"+str(day)
    else:
        day = str(day)
    hour = str(hour)
    if minute < 10:
        minute = "0"+str(minute)
    else:
        minute = str(minute)
    
    
    
    newsTimeStr = year+"."+month+"."+day+". "+ampm+" "+hour+":"+minute
                
                    

    return newsTimeStr


def getNextHourOfNewsTime(newsTimeStr):
    year = int(newsTimeStr[:4])
    month = int(newsTimeStr[5:7])
    day = int(newsTimeStr[8:10])
    ampm = newsTimeStr[12:14]
    if newsTimeStr[16] == ":":
        hour = int(newsTimeStr[15])
        minute = int(newsTimeStr[17:])
    else:
        hour = int(newsTimeStr[15:17])
        minute = int(newsTimeStr[18:])
            
            
    if int(hour) <= 10:
        hour = int(hour)+1
    else:
        if ampm == "오전" and hour == 12:
            hour = 1
        elif ampm == "오전" and hour == 11:
            ampm = "오후"
            hour = 12
        elif ampm == "오후" and hour == 12:
            hour =1
        else: #오후11시
            ampm = "오전"
            hour = 12
            nOfDays = howmanyDays(year, month)
            if int(day) < nOfDays:
                day = int(day)+1
            else:
                day = 1
                if month != 12:
                    month = int(month)+1
                else:
                    month = 1
                    year = int(year)+1
                        
    year = str(year)
    if month < 10:
        month = "0"+str(month)
    else:
        month = str(month)
    if day < 10:
        day = "0"+str(day)
    else:
        day = str(day)
    hour = str(hour)
    if minute < 10:
        minute = "0"+str(minute)
    else:
        minute = str(minute)
    
    
    
    newsTimeStr = year+"."+month+"."+day+". "+ampm+" "+hour+":"+minute
                
                    

    return newsTimeStr


def howmanyDays(year, month):
    isLeafYear = False
    if year%4 == 0 and year%100 != 0 or year % 400 == 0:
        isLeafYear = True
    
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    if month != 2:
        return 30
    if isLeafYear:
        return 29
    return 28

    
    
def fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, n, timeTerm, matrixIdx):
    _1minuteChartIdxTemp = _1minuteChartIdx
    newsPublishedTimeTradePrice = dataset1MinuteChart[_1minuteChartIdx][7]
    sumOfRiseFallRatio = 0
    #buySellIndicatorMatrix.append([])
    for i in range(0, n):
        _1minuteChartIdxTemp -= timeTerm                
        
        riseFallRatio = (dataset1MinuteChart[_1minuteChartIdxTemp][7] - newsPublishedTimeTradePrice)/newsPublishedTimeTradePrice
        sumOfRiseFallRatio += riseFallRatio
    avrgOfRiseFallRatio = sumOfRiseFallRatio / n
    buySellIndicatorMatrix[matrixIdx].append(avrgOfRiseFallRatio)      
    
    
def fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, n, timeTerm, matrixIdx):
    _1minuteChartIdxTemp = _1minuteChartIdx
    newsPublishedTimeTradePrice = dataset1MinuteChart[_1minuteChartIdx][7]
    countRise = 0
    countFall = 0
    #buySellIndicatorMatrix2.append([])
    for i in range(0, n):
        _1minuteChartIdxTemp -= timeTerm                
        
        riseFall = dataset1MinuteChart[_1minuteChartIdxTemp][7] - newsPublishedTimeTradePrice
        if riseFall >= 0:
            countRise += 1
        else:
            countFall += 1        
    
    if countRise >= countFall:
        riseFallCountRatio = countRise / n
        riseFallCountRatio *= 2
        riseFallCountRatio -= 1.0
    else:
        riseFallCountRatio = -(countFall / n)
        riseFallCountRatio *= 2
        riseFallCountRatio += 1.0
    
    buySellIndicatorMatrix2[matrixIdx].append(riseFallCountRatio) 
    
    
    
    newsSearchKeyword 
    
    
    
    
    
newsSearchKeyword = "비트코인"

newsStartYear = "2021"
newsStartMonth = "01"
newsEndYear = "2021"
newsEndMonth = "03"

#dfChart = pd.read_csv("1MinuteCandle_KRW-BTC_2022.01_2021.12.csv")
df1MinuteChart = pd.read_csv("1MinuteCandle_KRW-BTC_2022.01_2020.12.csv")
#dfNews = pd.read_csv("NewsSearchKeyword_비트코인_2021.12_2021.12.csv")
#dfNews = pd.read_csv("NewsSearchKeyword_비트코인_2021.7_2021.12.csv")
#dfNews = pd.read_csv("NewsSearchKeyword_비트코인_2021.1_2021.12.csv")
#dfNews = pd.read_csv("NewsSearchKeyword_비트코인_"+newsStartYear+"."+newsStartMonth+"_"+newsEndYear+"."+newsEndMonth+".csv")
dfNews = pd.read_csv("NewsSearchKeyword_"+newsSearchKeyword+"_"+newsStartYear+"."+newsStartMonth+"_"+newsEndYear+"."+newsEndMonth+".csv")
#dfMarketCapitalization = pd.read_csv("marketCapitalization_BTC_2022.01_2021.11.csv")
dfMarketCapitalization = pd.read_csv("marketCapitalization_BTC_2022.01_2020.12.csv")
df60MinuteChart = pd.read_csv("60MinuteCandle_KRW-BTC_2022.01_2020.12.csv")



nOfBuySellSuggestionSamples = 11
nOfTradingTimePredictSamples = 14







dataset1MinuteChart = df1MinuteChart.values
datasetNews = dfNews.values
datasetMarketCapitalization = dfMarketCapitalization.values
dataset60MinuteChart = df60MinuteChart.values
print("print(len(datasetNews))")
print(len(datasetNews))


_1minuteChartIdxConverter = {}
idx = 0
for dataset1MinuteChartRow in dataset1MinuteChart:
    newsTimeStr = fromChartTime2NewsTime(dataset1MinuteChartRow[3])
    _1minuteChartIdxConverter[newsTimeStr] = idx
    idx += 1
    
marketCapitalizationIdxConverter = {}
idx = 0
for datasetMarketCapitalizationRow in datasetMarketCapitalization:
    newsDateStr = fromMarketCapitalizationDateToNewsDate(datasetMarketCapitalizationRow[1])
    marketCapitalizationIdxConverter[newsDateStr] = idx
    idx += 1
    
_60minuteChartIdxConverter = {}
idx = 0
#보조지표 계산 과정에서 과거 데이터의 인덱스값을 작은 수로, 최신 데이터의 인덱스값을 큰 수로 함
for i in range(len(dataset60MinuteChart)-1, -1, -1):
    newsTimeStr = fromChartTime2NewsTime(dataset60MinuteChart[i][3])
    _60minuteChartIdxConverter[newsTimeStr] = idx
    idx += 1
    
    
    

stochasticJudgementList = []
MACDJudgementList = []
RSIJudgementList = []


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
#stochasticIdx = _60minuteChartIdxConverter["2021.11.30. 오후 6:00"]
stochasticIdx = _60minuteChartIdxConverter[newsStartYear+"."+newsStartMonth+"."+"01. 오전 12:00"]
stochasticIdx -= stochasticParameter[1] + stochasticParameter[2] - 2



coinPriceList_Stochastic = []

for i in range(stochasticIdx, len(dataset60MinuteChart)):
    # 패스트 스토캐스틱 %K 
    currentPrice = dataset60MinuteChart[i][7]
    NDaysLowestPrice = currentPrice
    NDaysHighestPrice = currentPrice    
    
    for j in range(1, stochasticParameter[0]+1):
        jDaysAgoLowPrice = dataset60MinuteChart[i-j][6]
        if jDaysAgoLowPrice < NDaysLowestPrice:
            NDaysLowestPrice = jDaysAgoLowPrice
        jDaysAgoHighPrice = dataset60MinuteChart[i-j][5]
        if jDaysAgoHighPrice > NDaysHighestPrice:
            NDaysHighestPrice = jDaysAgoHighPrice    
    
    fastStochasticPercentK = (currentPrice-NDaysLowestPrice)/(NDaysHighestPrice-NDaysLowestPrice)
    
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
    

#stochasticList = [패스트 %K, 패스트 %D, 슬로우 %K, 슬로우 %D]
for i in range(6, len(stochasticList)-1):      
    if( stochasticList[i][2] <= 0.20 and stochasticList[i][3] <= 0.20 ):
        if stochasticList[i-1][2] <= stochasticList[i-1][3] and stochasticList[i+1][2] >= stochasticList[i+1][3]:
            stochasticJudgementList.append(1)
            continue
    elif( stochasticList[i][2] >= 0.80 and stochasticList[i][3] >= 0.80 ):
        if stochasticList[i-1][2] >= stochasticList[i-1][3] and stochasticList[i+1][2] <= stochasticList[i+1][3]:
            stochasticJudgementList.append(-1)
            continue;
            
    stochasticJudgementList.append(0)


    
    
#MACD 보조지표
#parameter: (12, 26, 9)
#지수이동평균(EMA) = 금일 종가 * 승수 + 작일 EMA * 승수
#EMA 승수 : 2/(1+n)
#MACD = 단기 지수이동평균(n_0일) - 장기 지수이동평균(n_1일)
#signal : MACD의 n_2일 지수이동평균

#MACDList = [단기 지수이동평균, 장기 지수이동평균, MACD, signal]
MACDParameter = [12, 26, 9]
MACDList = []
#MACDIdx = _60minuteChartIdxConverter["2021.11.30. 오후 4:00"]
#MACDIdx = _60minuteChartIdxConverter["2021.11.30. 오후 3:00"]
MACDIdx = _60minuteChartIdxConverter[newsStartYear+"."+newsStartMonth+"."+"01. 오전 12:00"]
MACDIdx -= MACDParameter[2]

initialShortPeriodEMA = 0
initialLongPeriodEMA = 0


coinPriceList_MACD = []

for i in range(MACDIdx-(MACDParameter[0]-1) , MACDIdx+1):
    initialShortPeriodEMA += dataset60MinuteChart[i][7]
initialShortPeriodEMA /= MACDParameter[0]

for i in range(MACDIdx-(MACDParameter[1]-1) , MACDIdx+1):
    initialLongPeriodEMA += dataset60MinuteChart[i][7]
initialLongPeriodEMA /= MACDParameter[1]
MACDList.append([initialShortPeriodEMA, initialLongPeriodEMA, initialShortPeriodEMA-initialLongPeriodEMA, -1])
coinPriceList_MACD.append(dataset60MinuteChart[MACDIdx][7])

for i in range(MACDIdx+1, len(dataset60MinuteChart)):
    # 단기 지수이동평균, 장기 지수이동평균, MACD
    shortPeriodEP = 2/(MACDParameter[0]+1)
    longPeriodEP = 2/(MACDParameter[1]+1)   
    currentPrice = dataset60MinuteChart[i][7]
    shortPeriodEMA = currentPrice * shortPeriodEP + MACDList[-1][0] * (1-shortPeriodEP)
    longPeriodEMA = currentPrice * longPeriodEP + MACDList[-1][1] * (1-longPeriodEP)
    MACD = shortPeriodEMA - longPeriodEMA
    MACDList.append([shortPeriodEMA, longPeriodEMA, MACD, -1])
    
    coinPriceList_MACD.append(currentPrice)
    
    
for i in range(MACDParameter[2]-1, len(MACDList)):
    #signal
    
    movingAverage = MACDList[i][2]
    #for j in range(i-(MACDParameter[1]-1), i):
    for j in range(i-(MACDParameter[2]-1), i):
        #movingAverage += MACDList[j][0]
        movingAverage += MACDList[j][2]
    MACDList[i][3] = movingAverage/MACDParameter[2]


#MACDList = [단기 지수이동평균, 장기 지수이동평균, MACD, signal]
#for i in range(8, len(MACDList)-1):
MACDJudgementList.append(0)
#이동평균선 계산을 착각하여 이전 ipynb에서 8으로 하였으나 9가 맞음, 
#MACD 분석 과정에서 1일 전 데이터를 참조하므로 첫 항은 0으로 하고 10부터 시작 
for i in range(10, len(MACDList)-1):
    #print(MACDList[i-1][2], end="\t")
    #print(MACDList[i-1][3], end="\t")
    #print(MACDList[i+1][2], end="\t")
    #print(MACDList[i+1][3])
    if MACDList[i-1][2] <= MACDList[i-1][3] and MACDList[i+1][2] >= MACDList[i+1][3]:
        MACDJudgementList.append(1)
        continue
    elif MACDList[i-1][2] >= MACDList[i-1][3] and MACDList[i+1][2] <= MACDList[i+1][3]:
        MACDJudgementList.append(-1)
        continue
            
    MACDJudgementList.append(0)
  



#RSI 보조지표
#parameter: 14
#가격이 전일 가격보다 상승한 날의 상승분은 U(up) 값, 하락한 날의 하락분은 D(down) 값
#U값과 D값의 평균값은 각각 AU(average ups)와 AD(average downs) 값
#AU를 AD값으로 나눈 것을 RS(relative strength) 값 
#RSI = RS / (1 + RS)

#RSIList = [RSI]
RSIParameter = [14]
RSIList = []
#RSIIdx =  _60minuteChartIdxConverter["2021.12.01. 오전 12:00"]
RSIIdx =  _60minuteChartIdxConverter[newsStartYear+"."+newsStartMonth+"."+"01. 오전 12:00"]

diffList = [0]
#print(stochasticIdx)
#print(len(datasetDayChart))

for i in range(1, len(dataset60MinuteChart)):
    diffList.append(dataset60MinuteChart[i][7]-dataset60MinuteChart[i-1][7])   
    
      
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

    

for i in range(RSIIdx, len(dataset60MinuteChart)):
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
    coinPriceList_RSI.append(dataset60MinuteChart[i][7])
    



#RSIList = [RSI]
for i in range(0, len(RSIList)):
    if RSIList[i][0] >= 0.70:
        RSIJudgementList.append(-1)
    elif RSIList[i][0] <= 0.30:
        RSIJudgementList.append(1)
    else:
        RSIJudgementList.append(0)

    
    
        






newsTextSet = []
buySellIndicator=[]
tradingTimePredictSampleCategorizationList = []
buySellIndicatorMatrix=[]
buySellIndicatorMatrix2=[]
for i in range(0, 16):
    buySellIndicatorMatrix.append([])
    buySellIndicatorMatrix2.append([])
print("len(datasetNews)")
print(len(datasetNews))
for datasetNewsRow in datasetNews:
    isError = False
    while True:
        try:
            if datasetNewsRow[2] not in _1minuteChartIdxConverter:
                isError = True
                raise Exception("시세 데이터셋에 누락된 시간")                
            
            isError = False
            _1minuteChartIdx = _1minuteChartIdxConverter[datasetNewsRow[2]]
        except Exception as e:
            print("ERROR: ", e, end=" ")
            print(datasetNewsRow[2])
            datasetNewsRow[2] = getNextMinuteOfNewsTime(datasetNewsRow[2])
            
            
        if isError == False:
            break;
        
        
        
    
    #print("*************")
    #print(datasetNewsRow[2][:10])
    marketCapitalizationIdx = marketCapitalizationIdxConverter[datasetNewsRow[2][:10]]
    
    
 
    
    newsTextSet.append(datasetNewsRow[3])
    

    
    barometer = 0
    newsPublishedTimeTradePrice = dataset1MinuteChart[_1minuteChartIdx][7]
    
    tradingTimePredictCounter = 0
    tradingTimePredictSampleIdx = 0
    
    _1minuteChartIdxPrev = _1minuteChartIdxNow = _1minuteChartIdx    
    #chartIdxPrev = chartIdxNow = chartIdx
    marketCapitalizationIdxNow = marketCapitalizationIdx
    
    
    accumulatedTradeVolume = 0
    _1minuteChartIdxNow -= 1
    j = _1minuteChartIdxNow  
    marketCapitalizationPivot = datasetMarketCapitalization[marketCapitalizationIdx][2]
    accumulatedTradeVolume += dataset1MinuteChart[j][10] / marketCapitalizationPivot

    
    
    riseFallRatio = (dataset1MinuteChart[_1minuteChartIdx][7] - newsPublishedTimeTradePrice)/newsPublishedTimeTradePrice
    barometer += accumulatedTradeVolume * riseFallRatio
    
  
    _1minuteChartIdxPrev = _1minuteChartIdxNow
    
    

    for i in range(1, nOfBuySellSuggestionSamples): 
        _1minuteChartIdxNow -= 2 ** (i-1)
        #시가총액 및 발행량 데이터는 일단위이므로 1440으로 나누며 절사하여 n일전을 계산, 최대 23시간 59분의 오차 발생 가능
        marketCapitalizationIdxNow = marketCapitalizationIdx - int((2 ** i)/1440)
        for j in range(_1minuteChartIdxNow, _1minuteChartIdxPrev):
            marketCapitalizationNow = datasetMarketCapitalization[marketCapitalizationIdxNow][2]
            accumulatedTradeVolume += dataset1MinuteChart[j][10] / marketCapitalizationNow
            

        
        
        riseFallRatio = (dataset1MinuteChart[_1minuteChartIdxNow][7] - newsPublishedTimeTradePrice)/newsPublishedTimeTradePrice
        coefficient = 2 ** (-i)
        barometer += accumulatedTradeVolume * riseFallRatio * coefficient

    
        _1minuteChartIdxPrev = _1minuteChartIdxNow
        
        
    buySellIndicator.append(barometer)

    

    
    fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, 50, 1, 0)
    fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, 50, 5, 1)   
    fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, 50, 15, 2)
    fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, 50, 60, 3)
    fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, 50, 360, 4)
    fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, 50, 1440, 5)
    fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, 100, 1, 6)
    fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, 100, 5, 7)
    fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, 100, 15, 8)
    fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, 100, 60, 9)
    fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, 100, 360, 10)
    fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, 200, 1, 11)
    fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, 200, 5, 12)
    fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, 200, 15, 13)
    fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, 200, 60, 14)
    fillBuySellIndicatorMatrixRFRatioAvrg(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix, 200, 360, 15)
    
 
    

        
    
    fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, 50, 1, 0)
    fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, 50, 5, 1)
    fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, 50, 15, 2)
    fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, 50, 60, 3)
    fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, 50, 360, 4)
    fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, 50, 1440, 5)
    fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, 100, 1, 6)
    fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, 100, 5, 7)
    fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, 100, 15, 8)
    fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, 100, 60, 9)
    fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, 100, 360, 10)
    fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, 200, 1, 11)
    fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, 200, 5, 12)
    fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, 200, 15, 13)
    fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, 200, 60, 14)
    fillBuySellIndicatorMatrixRFCount(_1minuteChartIdx, dataset1MinuteChart, buySellIndicatorMatrix2, 200, 360, 15)

    
    
    

        

#print("*************************")
#for i in buySellIndicator:
#    print(i)




maxBuySellindicatorPositive = 0
maxBuySellindicatorNegative = 0
for i in buySellIndicator:
    if i >=0 :
        if i > maxBuySellindicatorPositive:
            maxBuySellindicatorPositive = i
    else:
        if i < maxBuySellindicatorNegative:
            maxBuySellindicatorNegative = i

        
for i in range(0, len(buySellIndicator)):
    if buySellIndicator[i] >= 0 :
        buySellIndicator[i] /= maxBuySellindicatorPositive
    else:
        buySellIndicator[i] /= abs(maxBuySellindicatorNegative)
    
#print("*************************")
#for i in buySellIndicator:
#    print(i)

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
        
    

buySellSuggestionMatrix=[]
for i in range(0, len(buySellIndicatorMatrix)):
    maxBuySellindicatorPositive = 0
    maxBuySellindicatorNegative = 0
    for j in buySellIndicatorMatrix[i]:
        if j >=0 :
            if j > maxBuySellindicatorPositive:
                maxBuySellindicatorPositive = j
        else:
            if j < maxBuySellindicatorNegative:
                maxBuySellindicatorNegative = j

    #print("***********")
    #print("maxBuySellindicator")
    #print(maxBuySellindicatorPositive)
    #print(maxBuySellindicatorNegative)


    for j in range(0, len(buySellIndicatorMatrix[i])):
        if buySellIndicatorMatrix[i][j] >= 0 :
            buySellIndicatorMatrix[i][j] /= maxBuySellindicatorPositive
        else:
            buySellIndicatorMatrix[i][j] /= abs(maxBuySellindicatorNegative)

    #print("*************************")
    #for i in buySellIndicator:
    #    print(i)

    
    buySellSuggestionMatrix.append([])
    for j in range(0, len(buySellIndicatorMatrix[i])):
        if buySellIndicatorMatrix[i][j] >= 0.7:
            buySellSuggestionMatrix[i].append(0)
        elif buySellIndicatorMatrix[i][j] >= 0.4:
            buySellSuggestionMatrix[i].append(1)
        elif buySellIndicatorMatrix[i][j] >= 0.1:
            buySellSuggestionMatrix[i].append(2)
        elif buySellIndicatorMatrix[i][j] >= -0.1:
            buySellSuggestionMatrix[i].append(3)
        elif buySellIndicatorMatrix[i][j] >= -0.4:
            buySellSuggestionMatrix[i].append(4)
        elif buySellIndicatorMatrix[i][j] >= -0.7:
            buySellSuggestionMatrix[i].append(5)
        else:
            buySellSuggestionMatrix[i].append(6)


buySellSuggestionMatrix2=[]
for i in range(0, len(buySellIndicatorMatrix2)):
    maxBuySellindicatorPositive = 0
    maxBuySellindicatorNegative = 0
    for j in buySellIndicatorMatrix2[i]:
        if j >=0 :
            if j > maxBuySellindicatorPositive:
                maxBuySellindicatorPositive = j
        else:
            if j < maxBuySellindicatorNegative:
                maxBuySellindicatorNegative = j

    #print("***********")
    #print("maxBuySellindicator")
    #print(maxBuySellindicatorPositive)
    #print(maxBuySellindicatorNegative)


    for j in range(0, len(buySellIndicatorMatrix2[i])):
        if buySellIndicatorMatrix2[i][j] >= 0 :
            buySellIndicatorMatrix2[i][j] /= maxBuySellindicatorPositive
        else:
            buySellIndicatorMatrix2[i][j] /= abs(maxBuySellindicatorNegative)

    #print("*************************")
    #for i in buySellIndicator:
    #    print(i)

    
    buySellSuggestionMatrix2.append([])
    for j in range(0, len(buySellIndicatorMatrix2[i])):
        if buySellIndicatorMatrix2[i][j] >= 0.7:
            buySellSuggestionMatrix2[i].append(0)
        elif buySellIndicatorMatrix2[i][j] >= 0.4:
            buySellSuggestionMatrix2[i].append(1)
        elif buySellIndicatorMatrix2[i][j] >= 0.1:
            buySellSuggestionMatrix2[i].append(2)
        elif buySellIndicatorMatrix2[i][j] >= -0.1:
            buySellSuggestionMatrix2[i].append(3)
        elif buySellIndicatorMatrix2[i][j] >= -0.4:
            buySellSuggestionMatrix2[i].append(4)
        elif buySellIndicatorMatrix2[i][j] >= -0.7:
            buySellSuggestionMatrix2[i].append(5)
        else:
            buySellSuggestionMatrix2[i].append(6)
     

            
            

#print("****************************")
#for i in buySellSuggestion:
#    print(i)




print("len(buySellSuggestion)")  
print(len(buySellSuggestion))

print("len(buySellSuggestionMatrix[4])")
print(len(buySellSuggestionMatrix[4]))

print("len(buySellSuggestionMatrix[10])")
print(len(buySellSuggestionMatrix[10]))

print("len(buySellSuggestionMatrix[15])")
print(len(buySellSuggestionMatrix[15]))

print("len(buySellSuggestionMatrix2[4])")
print(len(buySellSuggestionMatrix2[4]))

print("len(buySellSuggestionMatrix2[10])")
print(len(buySellSuggestionMatrix2[10]))

print("len(buySellSuggestionMatrix2[15])")
print(len(buySellSuggestionMatrix2[15]))

            
    
buySellSuggestionLevelAvrgList = []
buySellSuggestionLevelAvrgDistribution = [0, 0, 0, 0, 0, 0, 0]
for i in range(0, len(buySellSuggestion)):
    print([buySellSuggestion[i], buySellSuggestionMatrix[4][i], buySellSuggestionMatrix[10][i], buySellSuggestionMatrix[15][i],
           buySellSuggestionMatrix2[4][i], buySellSuggestionMatrix2[10][i], buySellSuggestionMatrix2[15][i]])
    levelAvrg = (buySellSuggestion[i]+buySellSuggestionMatrix[4][i]+buySellSuggestionMatrix[10][i]+buySellSuggestionMatrix[15][i]
           +buySellSuggestionMatrix2[4][i]+buySellSuggestionMatrix2[10][i]+buySellSuggestionMatrix2[15][i])/7
    print(levelAvrg," -> " , round(levelAvrg))
    
    buySellSuggestionLevelAvrgList.append(round(levelAvrg))
    buySellSuggestionLevelAvrgDistribution[round(levelAvrg)] += 1
    
print("buySellSuggestionLevelAvrgDistribution")
print(buySellSuggestionLevelAvrgDistribution)    

print("tradingTimePredictSampleCategorizationList")
tradingTimePredictSampleCategorizationList = []
TTPSCLidx=0
for datasetNewsRow in datasetNews:
    isError = False
    while True:
        try:
            if datasetNewsRow[2] not in _1minuteChartIdxConverter:
                isError = True
                raise Exception("시세 데이터셋에 누락된 시간")                
            
            isError = False
            _1minuteChartIdx = _1minuteChartIdxConverter[datasetNewsRow[2]]
        except Exception as e:
            print("ERROR: ", e, end=" ")
            print(datasetNewsRow[2])
            datasetNewsRow[2] = getNextMinuteOfNewsTime(datasetNewsRow[2])
            
            
        if isError == False:
            break;


    
    
    _60minuteNewsTime = ""
    if(len(datasetNewsRow[2]) == 19):
        # "2021.12.07. 오전 8:04"
        _60minuteNewsTime = datasetNewsRow[2][:17]
        _60minuteNewsTime += "00"
    else:
        #"2021.12.01. 오전 12:00"
        _60minuteNewsTime = datasetNewsRow[2][:18]
        _60minuteNewsTime += "00"
        
        
        
    isError = False
    while True:
        try:
            if _60minuteNewsTime not in _60minuteChartIdxConverter:
                isError = True
                raise Exception("시세 데이터셋에 누락된 시간")                
            
            isError = False
            _60minuteChartIdx = _60minuteChartIdxConverter[_60minuteNewsTime]
        except Exception as e:
            print("ERROR: ", e, end=" ")
            print(_60minuteNewsTime)
            _60minuteNewsTime = getNextHourOfNewsTime(_60minuteNewsTime)
            
            
        if isError == False:
            break;    
            
    #_60minuteChartIdx = _60minuteChartIdxConverter[_60minuteNewsTime]          
            

    
    judgementListIdx = _60minuteChartIdx+1 - _60minuteChartIdxConverter["2021.12.01. 오전 12:00"] 

    
    #tradingTimePredictSampleCategorizationList = []
    #judgementListIdx
    tradingTimePredictCounter = [0, 0, 0]
    tradingTimePredictSampleIdx = 1
    
    
    buySellSuggestionLevelAvrg = buySellSuggestionLevelAvrgList[TTPSCLidx]
    print("buySellSuggestionLevelAvrg ", buySellSuggestionLevelAvrg)
    TTPSCLidx += 1
    if buySellSuggestionLevelAvrg < 3:
        for i in range(judgementListIdx+1, len(stochasticJudgementList)):
            if stochasticJudgementList[i] == -1:
                tradingTimePredictCounter[0] = 1
            if MACDJudgementList[i] == -1:
                tradingTimePredictCounter[1] = 1
            if RSIJudgementList[i] == -1:
                tradingTimePredictCounter[2] = 1
            
            counterSum = 0
            for j in tradingTimePredictCounter:
                counterSum += j
            if counterSum >= 2:
                break;                
                
            tradingTimePredictSampleIdx += 1
            
        
        
    elif buySellSuggestionLevelAvrg > 3:
        for i in range(judgementListIdx+1, len(stochasticJudgementList)):
            if stochasticJudgementList[i] == 1:
                tradingTimePredictCounter[0] = 1
            if MACDJudgementList[i] == 1:
                tradingTimePredictCounter[1] = 1
            if RSIJudgementList[i] == 1:
                tradingTimePredictCounter[2] = 1
            
            counterSum = 0
            for j in tradingTimePredictCounter:
                counterSum += j
            if counterSum >= 2:
                break;                
                
            tradingTimePredictSampleIdx += 1
    
    else:
        tradingTimePredictSampleIdx = -1
        
    if tradingTimePredictSampleIdx == -1:
        tradingTimePredictSampleCategorizationList.append(-1)
    else:
        minuteDifference = tradingTimePredictSampleIdx * 60
        minuteDifferenceInPowOf2 = nOfTradingTimePredictSamples #아래 for문 끝까지 2로 나누어도 1이 되지 않는 경우 이 값(최대값)이 그대로 적용됨    
        for i in range(0, nOfTradingTimePredictSamples+1):
            minuteDifference = int(minuteDifference/2)
            if minuteDifference == 1:
                minuteDifferenceInPowOf2 = i                
                
        tradingTimePredictSampleCategorizationList.append(minuteDifferenceInPowOf2)
      
    


tradingTimePredictSampleCategorizationDistribution = []
for i in range(0, nOfTradingTimePredictSamples+2):
    tradingTimePredictSampleCategorizationDistribution.append(0)
    
print("len(tradingTimePredictSampleCategorizationList)")
print(len(tradingTimePredictSampleCategorizationList))
for i in tradingTimePredictSampleCategorizationList:
    print(i)
    tradingTimePredictSampleCategorizationDistribution[i+1] += 1
    
print("tradingTimePredictSampleCategorizationDistribution")
print(tradingTimePredictSampleCategorizationDistribution)



farthestTradingTimePredictSampleCategorization = 0
for i in range(0, len(tradingTimePredictSampleCategorizationDistribution)):
    if tradingTimePredictSampleCategorizationDistribution[i] > 0:
        farthestTradingTimePredictSampleCategorization = i
            
            

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
tradingTimePredictSampleCategorizationList_encoded = np_utils.to_categorical(tradingTimePredictSampleCategorizationList)
"""
print("*************20220302/1613**************")
print(len(buySellSuggestion))
print(len(buySellSuggestion_encoded))
print(len(tradingTimePredictSampleCategorizationList))
print(len(tradingTimePredictSampleCategorizationList_encoded))
"""

modelDir = "./models/"
if not os.path.exists(modelDir):
    os.mkdir(modelDir)



X_train0, X_test0, Y_train0, Y_test0 = train_test_split(padded_x, buySellSuggestion_encoded, test_size=0.3, random_state=seed)
#Y_train_encoded = np_utils.to_categorical(Y_train)
#Y_test_encoded = np_utils.to_categorical(Y_test)
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(padded_x, tradingTimePredictSampleCategorizationList_encoded, test_size=0.3, random_state=seed)


buySellSuggestionMatrix_encoded = []
XYTrainTestMatrix = []
for i in range(0, 16):
    buySellSuggestion_encoded_temp = np_utils.to_categorical(buySellSuggestionMatrix[i])
    X_train, X_test, Y_train, Y_test = train_test_split(padded_x, buySellSuggestion_encoded_temp, test_size=0.3, random_state=seed)
    XYTrainTestMatrix.append([X_train, X_test, Y_train, Y_test])

buySellSuggestionMatrix2_encoded = []
XYTrainTestMatrix2 = []
for i in range(0, 16):
    buySellSuggestion_encoded_temp = np_utils.to_categorical(buySellSuggestionMatrix2[i])
    X_train, X_test, Y_train, Y_test = train_test_split(padded_x, buySellSuggestion_encoded_temp, test_size=0.3, random_state=seed)
    XYTrainTestMatrix2.append([X_train, X_test, Y_train, Y_test])



"""
model0 = Sequential()
model0.add(Embedding(word_size, 256, input_length=x_maxLen))
#model0.add(Flatten())
#model.add(Dense(1, activation = 'sigmoid'))
model0.add(Dropout(0.25))
model0.add(Conv1D(32, kernel_size=5, padding="valid", activation="relu", strides=1))
model0.add(MaxPooling1D(pool_size=4))
model0.add(LSTM(256, activation='tanh'))
model0.add(Dense(7, activation='softmax'))

model0.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#print(X_train0.shape)
#print(X_test0.shape)
#print(Y_train0.shape)
#print(Y_test0.shape)



modelpath = modelDir+newsSearchKeyword+"_"+newsStartYear+"."+newsStartMonth+"_"+newsEndYear+"."+newsEndMonth+"_model0"+".hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_loss", verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor="val_loss", patience=7)



history = model0.fit(X_train0, Y_train0, epochs=20, batch_size=100, validation_data=(X_test0, Y_test0), callbacks=[early_stopping_callback, checkpointer])
del model0
print(modelpath)
model0 = load_model(modelpath)


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
"""


model1 = Sequential()
model1.add(Embedding(word_size, 256, input_length=x_maxLen))
#model.add(Flatten())
#model.add(Dense(1, activation = 'sigmoid'))
model1.add(Dropout(0.25))
model1.add(Conv1D(32, kernel_size=5, padding="valid", activation="relu", strides=1))
model1.add(MaxPooling1D(pool_size=4))
model1.add(LSTM(256, activation='tanh'))
model1.add(Dense(farthestTradingTimePredictSampleCategorization, activation='softmax'))

model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

modelpath = modelDir+newsSearchKeyword+"_"+newsStartYear+"."+newsStartMonth+"_"+newsEndYear+"."+newsEndMonth+"_model1_"+str(nOfTradingTimePredictSamples)+".hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_loss", verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5)



history = model1.fit(X_train1, Y_train1, epochs=20, batch_size=100, validation_data=(X_test1, Y_test1), callbacks=[early_stopping_callback, checkpointer])
del model1
print(modelpath)
model1 = load_model(modelpath)

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


"""
for i in range(0, 16):
    model = Sequential()
    model.add(Embedding(word_size, 256, input_length=x_maxLen))
    #model0.add(Flatten())
    #model.add(Dense(1, activation = 'sigmoid'))
    model.add(Dropout(0.25))
    model.add(Conv1D(32, kernel_size=5, padding="valid", activation="relu", strides=1))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(256, activation='tanh'))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #print(X_train0.shape)
    #print(X_test0.shape)
    #print(Y_train0.shape)
    #print(Y_test0.shape)
    
    modelpath = modelDir+newsSearchKeyword+"_"+newsStartYear+"."+newsStartMonth+"_"+newsEndYear+"."+newsEndMonth+"_model2_"+str(i)+".hdf5"
    checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_loss", verbose=1, save_best_only=True)

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=7)



    history = model.fit(XYTrainTestMatrix[i][0], XYTrainTestMatrix[i][2], epochs=20, batch_size=100, validation_data=(XYTrainTestMatrix[i][1], XYTrainTestMatrix[i][3]), callbacks=[early_stopping_callback, checkpointer])
    del model
    print(modelpath)
    model = load_model(modelpath)

    print("\n Test Accuracy: %.4f" % (model.evaluate(XYTrainTestMatrix[i][1], XYTrainTestMatrix[i][3])[1]))


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

    
for i in range(0, 16):
    model = Sequential()
    model.add(Embedding(word_size, 256, input_length=x_maxLen))
    #model0.add(Flatten())
    #model.add(Dense(1, activation = 'sigmoid'))
    model.add(Dropout(0.25))
    model.add(Conv1D(32, kernel_size=5, padding="valid", activation="relu", strides=1))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(256, activation='tanh'))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #print(X_train0.shape)
    #print(X_test0.shape)
    #print(Y_train0.shape)
    #print(Y_test0.shape)
    
    modelpath = modelDir+newsSearchKeyword+"_"+newsStartYear+"."+newsStartMonth+"_"+newsEndYear+"."+newsEndMonth+"_model3_"+str(i)+".hdf5"
    checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_loss", verbose=1, save_best_only=True)

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=7)



    history = model.fit(XYTrainTestMatrix2[i][0], XYTrainTestMatrix2[i][2], epochs=20, batch_size=100, validation_data=(XYTrainTestMatrix2[i][1], XYTrainTestMatrix2[i][3]), callbacks=[early_stopping_callback, checkpointer])
    del model
    print(modelpath)
    model = load_model(modelpath)

    print("\n Test Accuracy: %.4f" % (model.evaluate(XYTrainTestMatrix2[i][1], XYTrainTestMatrix2[i][3])[1]))


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
"""

