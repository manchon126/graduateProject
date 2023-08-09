import pandas as pd


df = pd.DataFrame([], columns=["data", "opening_price", "high_price", "low_price", "trade_price",
                                    "volume", "marketCapitalization"])
pdIdx=0   


f = open("./marketCapitalization.txt", 'r', encoding='UTF8')
while True:
    line = f.readline()
    if not line:
        break
    lineList = line.split('\t')
    


    new_pd_line = pd.DataFrame({"data":lineList[0], "opening_price":lineList[1], "high_price":lineList[2],
                                "low_price":lineList[3], "trade_price":lineList[4], "volume":lineList[5],
                                "marketCapitalization":lineList[6]}, index=[pdIdx])
    df = df.append(new_pd_line)
    pdIdx += 1




df.to_csv("marketCapitalization.csv", quoting=1)

f.close()