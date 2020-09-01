import sys

from SPARQLWrapper import SPARQLWrapper, JSON
import re

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.firefox import GeckoDriverManager
from bs4 import BeautifulSoup
import os
import requests
import urllib3
from urllib3.exceptions import InsecureRequestWarning
import time

patterns = {
    '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
    '[đ]': 'd',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    '[ỳýỷỹỵ]': 'y'
}


def convert(text):
    """
    Convert from 'Tieng Viet co dau' thanh 'Tieng Viet khong dau'
    text: input string to be converted
    Return: string converted
    """
    output = text
    for regex, replace in patterns.items():
        output = re.sub(regex, replace, output)
        # deal with upper case
        output = re.sub(regex.upper(), replace.upper(), output)
    return output


endpoint_url = "https://query.wikidata.org/sparql"

query = """SELECT ?person ?personLabel WHERE {
  SERVICE wikibase:label { bd:serviceParam wikibase:language "vi, en". }
  
  
  ?person wdt:P31 wd:Q5.
  ?person wdt:P27 wd:Q881.
  ?person wdt:P106 wd:Q177220.
  
}
LIMIT 150
"""


def get_results(endpoint_url, query):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


results = get_results(endpoint_url, query)
list_name_celeb = []
for result in results["results"]["bindings"]:
    list_name_celeb.append(result["personLabel"]["value"])

# from google_images_download import google_images_download
#
# response = google_images_download.googleimagesdownload()
#
# for x in list_name_celeb:
#     arguments = {"keywords":convert(x), "limit":50}
#     try:
#         absolute_image_paths = response.download(arguments)
#     except UnicodeDecodeError:
#         continue


urllib3.disable_warnings(InsecureRequestWarning)


geckodriver = '/home/lam/.wdm/drivers/geckodriver/linux64/v0.27.0/geckodriver'


def download_google_staticimages(pic_dirs, name):
    if not os.path.exists(pic_dirs):
        os.mkdir(pic_dirs)

    if not os.path.exists(os.path.join(pic_dirs, name)):
        os.mkdir(os.path.join(pic_dirs, name))

    searchurl = 'https://www.google.com/search?q=' + name.replace(' ', '+') + '&source=lnms&tbm=isch'
    options = webdriver.FirefoxOptions()
    options.add_argument('--no-sandbox')
    #options.add_argument('--headless')

    try:
        browser = webdriver.Firefox(executable_path=geckodriver, options=options)
    except Exception as e:
        print(f'No found chromedriver in this environment.')
        print(f'Install on your machine. exception: {e}')
        browser = webdriver.Firefox(GeckoDriverManager().install(), options=options)

    print(f'Getting you a lot of images. This may take a few moments...')

    browser.set_window_size(1280, 1024)
    browser.get(searchurl)
    time.sleep(1)

    # print(f'Getting you a lot of images. This may take a few moments...')
    #
    element = browser.find_element_by_tag_name('body')
    #
    # # Scroll down
    # #for i in range(30):
    for i in range(50):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)

    try:
        browser.find_element_by_id('smb').click()
        for i in range(50):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)
    except:
        for i in range(10):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)

    print(f'Reached end of page.')
    time.sleep(0.5)
    print(f'Retry')
    time.sleep(0.5)

    # # Below is in japanese "show more result" sentences. Change this word to your lanaguage if you require.
    # browser.find_element_by_xpath('//input[@value="Show more results"]').click()
    #
    # # Scroll down 2
    # for i in range(50):
    #     element.send_keys(Keys.PAGE_DOWN)
    #     time.sleep(0.3)
    #
    # try:
    #     browser.find_element_by_id('smb').click()
    #     for i in range(50):
    #         element.send_keys(Keys.PAGE_DOWN)
    #         time.sleep(0.3)
    # except:
    #     for i in range(10):
    #         element.send_keys(Keys.PAGE_DOWN)
    #         time.sleep(0.3)

    #elements = browser.find_elements_by_xpath('//div[@id="islrg"]')
    #page_source = elements[0].get_attribute('innerHTML')
    page_source = browser.page_source

    soup = BeautifulSoup(page_source, 'lxml')
    images = soup.find_all('img')

    urls = []
    for image in images:
        try:
            url = image['data-src']
            if not url.find('https://'):
                urls.append(url)
        except:
            try:
                url = image['src']
                if not url.find('https://'):
                    urls.append(image['src'])
            except Exception as e:
                print(f'No found image sources.')
                print(e)

    urls = urls[:50]
    count = 0
    if urls:
        for url in urls:
            try:
                res = requests.get(url, verify=False, stream=True)
                rawdata = res.raw.read()
                with open(os.path.join(pic_dirs, name, 'img_' + name + str(count) + '.jpg'), 'wb') as f:
                    f.write(rawdata)
                    count += 1
            except Exception as e:
                print('Failed to write rawdata.')
                print(e)

    browser.close()
    return count

for x in list_name_celeb:
    download_google_staticimages('pictures', x)