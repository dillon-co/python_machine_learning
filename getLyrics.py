# from urlparse import urljoin
# from bs4 import BeautifulSoup
# import requests
#
#
# BASE_URL = "http://genius.com"
# artist_url = "http://genius.com/artists/Andre-3000/"
#
# response = requests.get(artist_url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36'})
# print(response)
# soup = BeautifulSoup(response.text, "lxml")
# # print(soup)
# for song_link in soup.select('ul.song_list > li > a'):
#     print(song_link)
#     link = urljoin(BASE_URL, song_link['href'])
#     response = requests.get(link)
#     soup = BeautifulSoup(response.text)
#     lyrics = soup.find('div', class_='lyrics').text.strip()
#     print(lyrics)
#     text_file = open("eminem.txt", "w")
#     text_file.write("\n\n", lyrics)
#     text_file.close()
#     # tokenize `lyrics` with nltk

import re
import requests
from bs4 import BeautifulSoup

url = 'http://www.lyrics.com/eminem'
r = requests.get(url)
soup = BeautifulSoup(r.content, "lxml")
gdata = soup.find_all('div',{'class':'row'})
print len(gdata)
eminemLyrics = []

for item in gdata:
    title = item.find_all('a',{'itemprop':'name'})[0].text
    lyricsdotcom = 'http://www.lyrics.com'
    for link in item('a'):
        try:
            lyriclink = lyricsdotcom+link.get('href')
            req = requests.get(lyriclink)
            lyricsoup = BeautifulSoup(req.content)
            lyricdata = lyricsoup.find_all('div',{'id':re.compile('lyric_space|lyrics')})[0].text
            eminemLyrics.append(lyricdata)
            print title
            print lyricdata
            text_file = open("eminem.txt", "w")
            text_file.write("\n\n", eminmeLyrics)
            text_file.close()
        except:
            pass
