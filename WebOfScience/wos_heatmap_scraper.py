from selenium import webdriver
import pandas as pd

def heatmap_collection():

    #Initialize driver
    driver = webdriver.Chrome('D:/Documents/selenium/chromedriver_win32_79\chromedriver.exe')

    #Initialize keywords from csv files
    keyword_A = list(pd.read_csv('keywords_A.csv')['keywords'].apply(lambda x: str(x).replace('\xa0', '')))
    keyword_B = list(pd.read_csv('keywords_B.csv')['keywords'].apply(lambda x: str(x).replace('\xa0', '')))

    #Creating Dataframe that will hold the search results
    df = pd.DataFrame(columns=['keywordA', 'keywordB', 'number_of_recs'])

    #Automated download sequence using Selenium
    for kA in keyword_A:
        for kB in keyword_B:
                counter = 1
                driver.get(
                    "http://apps.webofknowledge.com.proxy.library.nyu.edu/WOS_GeneralSearch_input.do?product=WOS&search_mode=GeneralSearch&SID=8Ff9hT327EKMH8ak5zR&preferencesSaved=")
                driver.find_element_by_link_text("Advanced Search").click()

                driver.find_element_by_id("selectallTop").click()
                driver.find_element_by_id("deleteTop").click()

                driver.find_element_by_id("value(input1)").clear()

                search_term = 'TS=("' + kA + '" AND "' + kB + '")' + \
                              ' NOT TS=(' + '"Traumatic brain injury"' + ' OR ' + '"post traumatic stress disorder")'
                driver.find_element_by_id("value(input1)").send_keys(search_term)
                driver.find_element_by_id("search-button").click()
                try:
                    driver.implicitly_wait(0)
                    number_of_records = int(str(driver.find_element_by_id("hitCount").text).replace(',', ''))
                    df1 = pd.DataFrame({'keywordA': [kA], 'keywordB': [kB], 'number_of_recs': [number_of_records]})
                    df = df.append(df1, ignore_index=True)
                except:
                    number_of_records = 0
                    df1 = pd.DataFrame({'keywordA': [kA], 'keywordB': [kB], 'number_of_recs': [number_of_records]})
                    df = df.append(df1, ignore_index=True)

    return(df)

if __name__ == '__main__':
    df = heatmap_collection()
    df.to_csv('wos_heatmap_data.csv')