# -*- coding: utf-8 -*-
from telnetlib import EC
import selenium
from selenium import webdriver
import time
import math
import io
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
import pandas as pd
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from datetime import datetime

Keyfile = io.open("keys-a.txt","r",encoding="utf-8")
keywords = Keyfile.read().splitlines()
BioFile = io.open("keys-b.txt","r",encoding="utf-8")
biomarkersAll = BioFile.read().splitlines()

errors = open("Errors.txt","a+")

def main() :

    end = 5
    start = 0

    while (end < len(biomarkersAll)):

        total_elements = 0
        chrome_options = webdriver.ChromeOptions()
        prefs = {"profile.managed_default_content_settings.images": 2}
        chrome_options.add_argument("--no-proxy-server")
        chrome_options.add_argument("--proxy-server='direct://'")
        chrome_options.add_argument("--proxy-bypass-list=*")
        chrome_options.add_experimental_option("prefs", prefs)
        driver = webdriver.Chrome(options=chrome_options)
        base_url = 'https://www.biorxiv.org'
        search_url = 'https://www.biorxiv.org/search'

        biomarkers = biomarkersAll[start:end]
        row_list = []

        for biomarker in biomarkers:

            for keyword in keywords:

                driver.get(search_url)
                driver.find_element_by_id("edit-abstract-title").send_keys(biomarker + " " + keyword)
                driver.find_element_by_id("edit-text-abstract-title").send_keys(biomarker + " " + keyword)
                select = Select(driver.find_element_by_id('edit-numresults'))
                driver.find_element_by_id("edit-submit").send_keys(u'\ue007')
                num_of_results = driver.find_element_by_id("page-title").text.split(" ")[0]

                if (num_of_results != "No"):

                    num = int(num_of_results.replace(",", ""))
                    total_elements += num
                    checked = 0
                    keywordPair = biomarker + " " + keyword

                    pages = 1
                    Totalpages = math.ceil(num/10.0)

                    while (pages <= Totalpages):

                        i = 0
                        if (num < 10):
                            resultsOnPage = num
                        elif (pages != Totalpages):
                            resultsOnPage = 10
                        elif (pages == Totalpages):
                            resultsOnPage = num - (pages * 10)

                        for currArticle in range(1, resultsOnPage + 1):

                            # reset articles after driver has gone to new windows
                            try:
                                article = driver.find_element_by_xpath("//*[@id='hw-advance-search-result']/div/div/ul/li[" + str(currArticle) + "]")
                                # names = WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located((By.XPATH, ".//div/div/div/span")))
                                names = driver.find_elements_by_xpath(".//div/div/div/span")

                                # get the first five names
                                if (len(names)) > 5:
                                    names = names[0:5]
                                authors = ""

                                for name in names:
                                    #  since first, middle, and last stored seperately
                                    try:
                                        author = name.find_element_by_xpath(
                                            ".//span[1]").text + " " + name.find_element_by_xpath(
                                            ".//span[2]").text
                                        authors += ", " + author

                                    except NoSuchElementException:
                                        author = ""

                                authors = authors[2:]

                                # get title
                                try:
                                    title = article.find_element_by_xpath(".//div/div/span/a/span").text

                                except NoSuchElementException:
                                    title = "COULD NOT BE RETRIEVED"

                                # get doi
                                link = WebDriverWait(article, 10).until(
                                    EC.element_to_be_clickable((By.XPATH, ".//div/div/span/a"))
                                )
                                webdriver.ActionChains(driver).move_to_element(link).click(link).perform()

                                try:
                                    date = driver.find_element_by_xpath(
                                        "//*[@id='block-system-main']/div/div/div/div/div[2]/div/div/div[3]/div").text
                                    d = getDOI(date)

                                except NoSuchElementException:
                                    d = "COULD NOT BE RETRIEVED"

                                try:
                                    pdf_link = driver.find_element_by_xpath(
                                        "//*[@id='mini-panel-biorxiv_art_tools']/div/div[1]/div/div/div[1]/div/a").get_attribute(
                                        "href")

                                except NoSuchElementException:
                                    pdf_link = "COULD NOT BE RETRIEVED"

                                driver.back()

                                row = {'Title': title, 'DOI': d, 'Keyword Pair': keywordPair, 'Authors': authors,
                                       'PDF': pdf_link}
                                row_list.append(row)

                            # if it cant locate element (shouldnt happen but sometimes does if computer
                            #  is asleep for too long)
                            except NoSuchElementException:
                                # catalog to get the data later
                                print("error")
                                errors.write(str(checked) + " " + keywordPair)

                            # if cant find it in the assigned time
                            except TimeoutException:
                                # catalog for later
                                print("error")
                                errors.write(str(checked) + " " + keywordPair)

                            #  add the new data and increase counters
                            checked += 1
                            i += 1
                            pages += 1


                        # going to the next page
                        if (pages < Totalpages):

                            if checked == 10:

                                next = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                                    (By.XPATH, "//*[@id='hw-advance-search-result']/div/div/div/div/ul/li/a")))
                                webdriver.ActionChains(driver).move_to_element(next).click(next).perform()
                                driver.delete_all_cookies()

                            else:
                                next = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH,"//*[@id='hw-advance-search-result']/div/div/div/div[2]/ul/li/a")))
                                next.click()
                                driver.delete_all_cookies()

        # save to csv
        df = pd.DataFrame(row_list, columns=["Title", "DOI", "Keyword Pair", "Authors", "PDF"])
        df.to_csv('BioRXResults'+str(start)+'.csv', index=True, header=True, sep=',')
        driver.close()
        errors.close()
        print("Total Should Be Checked for " + str(start) + " : " + str(total_elements))

        if ( end == (len(biomarkersAll)-1)):
            break

        if ( end + 5 < (len(biomarkersAll)-1)):
            end += 5
            start = end - 4
        else:
            start = end
            end = (len(biomarkersAll)-1)



#  convert date into a DOI
def getDOI(date):

    date = date[7:]
    date = date.replace(',','')
    date = date.replace('.','')
    Month = ['January', 'February','March','April','May','June','July','August','September','October','November','December']
    sep = date.split()
    month = sep[0]
    day = sep[1]
    year = sep[2]
    fmonth = (Month.index(month) + 1)

    if int(fmonth) < 10:
        fmonth = "0" + str(fmonth)

    final_doi = str(year) + "-" + str(fmonth) + "-" + str(day)

    return final_doi


if __name__ == "__main__" :

    main()
