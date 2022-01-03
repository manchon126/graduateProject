#일간 데이터 수집, .csv로 변환
# 2017-09-25 ~ 2021-12-27

import requests
import pandas as pd

csvNum = 8
#url = "https://api.upbit.com/v1/candles/days?market=KRW-BTC&count=200"
url = "https://api.upbit.com/v1/candles/days?market=KRW-BTC&to=2018-02-27 00:00:00&count=200"
headers = {"Accept": "application/json"}
response = requests.request("GET", url, headers=headers)
responseTxt = response.text


df = pd.DataFrame([], columns=["market", "candle_date_time_utc",
                               "candle_date_time_kst", "opening_price",
                               "high_price", "low_price", "trade_price",
                               "timestamp", "candle_acc_trade_price",
                               "candle_acc_trade_volume", "prev_closing_price",
                               "change_price", "change_rate"])


textFlag = 0
pdIdx=200 * (csvNum-1)
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
                                   new_data[16]:new_data[17], new_data[18]:new_data[19], new_data[20]:new_data[21], new_data[22]:new_data[23],
                                   new_data[24]:new_data[25]}, index=[pdIdx])
        pdIdx += 1
        df = df.append(new_pd_line)     
    
        
        
df.to_csv("dayCandle_"+str(csvNum)+".csv", quoting=1)