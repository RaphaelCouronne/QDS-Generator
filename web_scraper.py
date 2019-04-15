import json
import os
import numpy as np
from urllib.request import urlopen
import bs4 as BeautifulSoup
import unidecode
import lyricsgenius

with open("secret.json") as json_file:
    secret = json.load(json_file)


#%% Get rappers
html = urlopen('https://fr.wikipedia.org/wiki/Cat%C3%A9gorie:Rappeur_fran%C3%A7ais').read()

with open("rappeurs.html") as html:
    soup = BeautifulSoup.BeautifulSoup(html, "lxml")
soup = soup.find_all('li')


def filter_rapper_name(name):

    suffix_labels = [' (rappeur)', " (artiste)",
                     " (rappeur français)", "/Brouillon",
                     ' (chanteur)', ' (musicien)']
    for suffix_label in suffix_labels:
        name = name.split(suffix_label)[0]

    prefix_labels = ['Utilisateur:', 'MarkHunt/']
    for prefix_label in prefix_labels:
        if len(name.split(prefix_label)) > 1:
            name = name.split(prefix_label)[1]

    return name


artists_name_scraped = []
for line in soup:
    artists_name_scraped.append(unidecode.unidecode(filter_rapper_name(line.find_all('a')[0].contents[0]).lower()))

#%%
genius = lyricsgenius.Genius(secret['client_access_token'])
genius._SLEEP_MIN = 0.2
genius.sleep_time = 0.2


artists_done = ['disiz la peste', 'nekfeu']

artists_name_written = ['lomepal', 'iam', 'romeo elvis', 'pnl', 'jul', 'vald', 'damso', 'niska', 'booba', 'kaaris', 'la fouine', 'lorenzo',
                        'kalash criminel', 'gradur', 'lacrim', 'mhd', 'casseurs flowters', 'dosseh', 'orelsan']

artists_name = np.unique(artists_name_written+artists_name_scraped)

#%%
# Create folder data if it does not exist
if not os.path.exists("data"):
    print("Creating folder data")
    os.mkdir("data")

for artist_name in artists_name:
    artist = genius.search_artist(artist_name, max_songs=500, sort="title")

    for song in artist.songs:
        if not os.path.exists(os.path.join("data", artist.name)):
            print("Creating folder data/" + artist.name)
            os.mkdir(os.path.join("data", artist.name))

        song.save_lyrics(os.path.join("data", artist.name, song.title), overwrite='y')


"""
base_url = "http://api.genius.com"
headers = {'Authorization': "Bearer "+secret['client_access_token']}
#headers = {'Authorization': secret['client_secret']}
search_url = base_url + "/search"
song_title = "Back"
#artist_name = "Alkpote"
data = {'q': song_title}
response = requests.get(search_url, data=data, headers=headers)
"""
