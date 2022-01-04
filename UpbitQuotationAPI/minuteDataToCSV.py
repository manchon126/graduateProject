#n분봉 데이터 수집, .csv로 변환


import requests
from time import sleep
import pandas as pd

#UpbitQuotationAPI의 marketCode.py에서 marketCode 열람
marketCode = "KRW-EOS"

#조회할 분봉 데이터의 가장 최근 연/월(31일 23:59:00 이 기준)
startYear=2021
startMonth="12"

#조회할 분봉 데이터의 가장 오래된 연/월(01일 00:00:00 이 기준)
endYear=2021
endMonth="01"


# 1, 3, 5, 15, 10, 30, 60, 240분봉 중 택1
minuteInterval = 1





year = startYear
month = startMonth


url = "https://api.upbit.com/v1/candles/minutes/"+str(minuteInterval)+"?market="+marketCode+"&to="+str(year)+"-"+str(month)+"-31 23:59:59&count=200"
headers = {"Accept": "application/json"}
response = requests.request("GET", url, headers=headers)
responseTxt = response.text


df = pd.DataFrame([], columns=["market", "candle_date_time_utc", "candle_date_time_kst", 
                               "opening_price", "high_price", "low_price", "trade_price", 
                               "timestamp", "candle_acc_trade_price", "candle_acc_trade_volume", "unit"])
                        
pdIdx = 0
#a=0
newMonth = "12"
newDay = ""
newHour = ""


newMinute = ""
monthDoneFlag = False
queryDoneFlag = False


monthDf = pd.DataFrame([], columns=["market", "candle_date_time_utc", "candle_date_time_kst",
                                    "opening_price", "high_price", "low_price", "trade_price",
                                    "timestamp", "candle_acc_trade_price", "candle_acc_trade_volume", "unit"])

while True:
    #print(responseTxt)
    textFlag = 0
    for i in responseTxt:
        if i == '"':
            continue;

        if i =='{':
            new_data = []
            dayDataText = []
            textFlag = 1        
        elif textFlag==1 and i!=':':
            dayDataText.append(i)
        elif textFlag==1 and i==':':
            dayDataText = "".join(dayDataText)
            new_data.append(dayDataText)
            dayDataText = []
            textFlag=2
        elif textFlag==2 and i!=',' and i!='}':
             dayDataText.append(i)
        elif textFlag==2 and i==',':
            dayDataText = "".join(dayDataText)
            new_data.append(dayDataText)
            dayDataText = []
            textFlag=1
        elif i=='}':
            dayDataText = "".join(dayDataText)
            new_data.append(dayDataText)
            textFlag=0

            new_pd_line = pd.DataFrame({new_data[0]:new_data[1], new_data[2]:new_data[3], new_data[4]:new_data[5], new_data[6]:new_data[7], 
                                       new_data[8]:new_data[9], new_data[10]:new_data[11], new_data[12]:new_data[13], new_data[14]:new_data[15],
                                       new_data[16]:new_data[17], new_data[18]:new_data[19], new_data[20]:new_data[21]}, index=[pdIdx])
            pdIdx += 1
            monthDf = monthDf.append(new_pd_line)

            candle_date_time_UTC = new_data[3]
            #newYear = candle_date_time_UTC[:4]
            #newMonth = candle_date_time_UTC[5:7]
            newDay = candle_date_time_UTC[8:10]
            newHour = candle_date_time_UTC[11:13]            
            newMinute = candle_date_time_UTC[14:16]            
            #newSecond = candle_date_time_UTC[17:]

            #print(str(year)+"-"+str(month)+"-"+newDay+" "+newHour+":"+newMinute+":00")
            print(str(year)+"-"+newMonth+"-"+newDay+" "+newHour+":"+newMinute+":00")
            
            
            if year == endYear and newMonth ==endMonth and newDay=="01" and newHour=="00" and newMinute=="00":
                queryDoneFlag = True
                break;
                
                
            if newDay=="01" and newHour=="00" and newMinute=="00":
                monthDoneFlag = True                
                break;
                
            
              

    if monthDoneFlag:
        df = df.append(monthDf)
        monthDf = pd.DataFrame([], columns=["market", "candle_date_time_utc", "candle_date_time_kst",
                                            "opening_price","high_price", "low_price", "trade_price",
                                            "timestamp", "candle_acc_trade_price", "candle_acc_trade_volume", "unit"])                         


        newMonth = candle_date_time_UTC[5:7]
        if newMonth == "10":
            newMonth = "09"
        else:
            newMonth = newMonth[0] + str(int(newMonth[1])-1)

        monthDoneFlag = False
        
        if newMonth=="00":
            year = year-1            
            newMonth = "12"
            newDay = "31"
            newHour = "23"
            newMinute = "59"
            
                
     
    if queryDoneFlag:
        break;     
            


    sleep(0.11)
    url = "https://api.upbit.com/v1/candles/minutes/"+str(minuteInterval)+"?market="+marketCode+"&to="+str(year)+"-"+newMonth+"-"+newDay+" "+newHour+":"+newMinute+":00&count=200"
    headers = {"Accept": "application/json"}
    response = requests.request("GET", url, headers=headers)
    responseTxt = response.text
   
    #그 해 1월 1일 이후에 거래소에 상장한 경우
    if responseTxt== "[]": 
        break;
    
    
df.to_csv(str(minuteInterval)+"MinuteCandle_"+marketCode+"_"+str(year)+".csv", quoting=1)
