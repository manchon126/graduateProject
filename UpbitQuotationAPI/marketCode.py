import requests

#거래 가능한 종목들의 코드 조회 
#출력 형태:
#"market": "KRW-BTC",             (업비트 내에서 취급하는 각 종목별 코드)
#"korean_name": "비트코인",
#"english_name": "Bitcoin"
#"market_warning": "NONE"         (투자유의 종목으로 분류됐는가 여부, 출력생략가능)
url = "https://api.upbit.com/v1/market/all?isDetails=false"
headers = {"Accept": "application/json"}
response = requests.request("GET", url, headers=headers)
responseTxt = response.text

for i in responseTxt:
    if i==',':
        print(i, end="\n")
    else:
        print(i, end="")