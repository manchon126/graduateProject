#n분봉 데이터 수집, .csv로 변환
# 현재 2021-01-01 00:00:00 ~ 2021-12-27 23:59:00 수집 후 월별로 .csv파일 작성


import requests
from time import sleep
import pandas as pd


year=2021
month="12"

minuteInterval = 1


url = "https://api.upbit.com/v1/candles/minutes/"+str(minuteInterval)+"?market=KRW-BTC&to="+str(year)+"-"+str(month)+"-31 23:59:59&count=200"
headers = {"Accept": "application/json"}
response = requests.request("GET", url, headers=headers)
responseTxt = response.text


df = pd.DataFrame([], columns=["market", "candle_date_time_utc",
                                   "candle_date_time_kst", "opening_price",
                                   "high_price", "low_price", "trade_price",
                                   "timestamp", "candle_acc_trade_price",
                                   "candle_acc_trade_volume", "unit"])
pdIdx = 0
#a=0
newDay = ""
newHour = ""         
newMinute = ""
monthDoneFlag = False
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
            df = df.append(new_pd_line)
            
            candle_date_time_UTC = new_data[3]
            #newYear = candle_date_time_UTC[:4]
            #newMonth = candle_date_time_UTC[5:7]
            newDay = candle_date_time_UTC[8:10]
            newHour = candle_date_time_UTC[11:13]            
            newMinute = candle_date_time_UTC[14:16]            
            #newSecond = candle_date_time_UTC[17:]
            
            print(str(year)+"-"+str(month)+"-"+newDay+" "+newHour+":"+newMinute+":00")
            if newDay=="01" and newHour=="00" and newMinute=="00":
                monthDoneFlag = True
                break;
            
   
    if monthDoneFlag:
        break
    
    
    sleep(0.11)
    url = "https://api.upbit.com/v1/candles/minutes/"+str(minuteInterval)+"?market=KRW-BTC&to="+str(year)+"-"+str(month)+"-"+newDay+" "+newHour+":"+newMinute+":00&count=200"
    headers = {"Accept": "application/json"}
    response = requests.request("GET", url, headers=headers)
    responseTxt = response.text
    
        
        
        
        
df.to_csv("MinuteCandle_"+str(year)+"_"+str(month)+".csv", quoting=1)