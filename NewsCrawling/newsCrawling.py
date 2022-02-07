from selenium import webdriver
import time
from bs4 import BeautifulSoup
import datetime
from datetime import timedelta
import pandas as pd



chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome('chromedriver', chrome_options=chrome_options)
#driver = webdriver.Chrome()
#url = "https://kr.investing.com/news/cryptocurrency-news/article-740246"

searchKeyword = "비트코인"

startYear = 2021
startMonth = 1

endYear = 2021
endMonth = 12





newsDf = pd.DataFrame([], columns=["newsTitle", "newsTime", "newsContent"])
pdIdx = 0



d = datetime.datetime(startYear, startMonth, 1)
if endMonth == 12:
    tempEndYear = endYear+1
    tempEndMonth = 1
else:
    tempEndYear = endYear
    tempEndMonth = endMonth+1
ed = datetime.datetime(tempEndYear, tempEndMonth, 1) - timedelta(days=1)

while (ed - d).days >= 7:
    weekStartYear = d.year
    weekStartMonth = d.month
    if weekStartMonth < 10:
        weekStartMonthStr = "0" + str(weekStartMonth)
    else:
        weekStartMonthStr = str(weekStartMonth)
    weekStartDay = d.day
    if weekStartDay < 10:
        weekStartDayStr = "0" + str(weekStartDay)
    else:
        weekStartDayStr = str(weekStartDay)
        
    d = d + timedelta(days=6)
    
    weekEndYear = d.year
    weekEndMonth = d.month
    if weekEndMonth < 10:
        weekEndMonthStr = "0" + str(weekEndMonth)
    else:
        weekEndMonthStr = str(weekEndMonth)
    weekEndDay = d.day
    if weekEndDay < 10:
        weekEndDayStr = "0" + str(weekEndDay)
    else:
        weekEndDayStr = str(weekEndDay)

    d = d + timedelta(days=1)
    
    url = "https://search.naver.com/search.naver?where=news&query="+searchKeyword+"&sm=tab_opt&sort=0&photo=0&field=0&pd=3&ds="+str(weekStartYear)+"."+weekStartMonthStr+"."+weekStartDayStr+"&de="+str(weekEndYear)+"."+weekEndMonthStr+"."+weekEndDayStr+"&docid=&related=0&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so%3Ar%2Cp%3Afrom"+str(weekStartYear)+weekStartMonthStr+weekStartDayStr+"to"+str(weekEndYear)+weekEndMonthStr+weekEndDayStr+"&is_sug_officeid=0"
    print(url)
    driver.get(url)

    time.sleep(5)

    for i in range(0, 5):
        html = driver.page_source

        soup = BeautifulSoup(html, 'html.parser')

        try:
            newsList = soup.select_one(".list_news")
            newsLinkList = newsList.select("li > div > div > div > div > a")
            #print(newsList)
        except:
            print(str(weekStartYear)+"."+weekStartMonthStr+"."+weekStartDayStr+"~"+str(weekEndYear)+"."+weekEndMonthStr+"."+weekEndDayStr, end=": ")
            print("News does not exist")
            break

        newsLinks = []
        for newslink in newsLinkList:
            link = newslink['href']
            if link[:23] == "https://news.naver.com/":
                newsLinks.append(link)


        print(newsLinks)

        for newsLink in newsLinks:
            driver.get(newsLink)

            time.sleep(2)


            html = driver.page_source


            soup = BeautifulSoup(html, 'html.parser')

            try:
                newsTitle = soup.select_one("#articleTitle")
                newsTitle = newsTitle.text

                newsTime = soup.select_one(".t11")
                newsTime = newsTime.text

                newsContent = soup.select_one("#articleBodyContents")
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


            pd_news_text_line = pd.DataFrame({"newsTitle":newsTitle, "newsTime":newsTime, "newsContent":newsContent}, 
                                        index=[pdIdx])
            pdIdx = pdIdx+1
            newsDf = newsDf.append(pd_news_text_line)
            
            
            
        
        driver.get(url) #"""각 뉴스 페이지에서 다시 검색어-검색기간에 따른 뉴스목록 페이지로 돌아가기"""
        try:
            x_path = "//*[@id=\"main_pack\"]/div[2]/div/div/a["+str(i+2)+"]"
            nextNumPageButton = driver.find_element_by_xpath(x_path)

            nextNumPageButton.click()
        except:
            if i+2 < 6:
                print(str(weekStartYear)+"."+weekStartMonthStr+"."+weekStartDayStr+"~"+str(weekEndYear)+"."+weekEndMonthStr+"."+weekEndDayStr, end=": ")
                print("News list page "+str(i+2)+" does not exist")
            break;
        
        
        
            
        
        
print(newsDf)
  
newsDf.to_csv("NewsSearchKeyword"+"_"+searchKeyword+"_"+str(startYear)+"."+str(startMonth)+"_"+str(endYear)+"."+str(endMonth)+".csv", quoting=1)
