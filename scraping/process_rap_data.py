
import os

folder = "data/french_rap"
artist_folder = os.path.join(folder, 'booba')

#%%
os.listdir(folder)



#%%
import json

lyrics_rap = []

for artist_name in os.listdir(folder):
    artist_folder = os.path.join(folder, artist_name)

    lyrics_artist = []

    for i, song_name in enumerate(os.listdir(artist_folder)):

        song_path = os.path.join(artist_folder,song_name)
        with open(song_path) as json_file:
            data = json.load(json_file)
        lyrics = data['songs'][0]['lyrics']

        # TODO \n, ', couplet

        lyrics = lyrics.replace("'", "' ")
        lyrics = lyrics.replace("\n", " \n ")
        lyrics = lyrics.replace("Refrain:", "")
        lyrics = lyrics.replace(": ", "")
        lyrics = lyrics.replace(" x2 ", "")
        lyrics = lyrics.replace(" x4 ", "")
        lyrics = lyrics.replace(" x8 ", "")
        lyrics = lyrics.replace(" Paroles rédigées et expliquées par la communauté Rap Genius France !", "")
        lyrics = lyrics.replace(" Paroles rédigées et expliquées par la communauté Rap Genius France", "")
        lyrics = "<BeginSong> " + lyrics + " <EndSong>"
        lyrics = "\n == \n" + artist_name + "\n == \n" + lyrics

        import re
        lyrics = re.sub("[\(\[].*?[\)\]]", "", lyrics)

        lyrics_artist.append(lyrics)
        #if i>0:
        #    break
    lyrics_rap.append(" ".join(lyrics_artist))

#%%
#booba_lyrics = " ".join(lyrics_artist)

#with open("data/processed_rap/booba.txt", 'w') as file:
#    file.writelines(booba_lyrics)


#%% Lyrics all

concatenated_rap_lyrics = " ".join(lyrics_rap)

with open("data/processed_rap/french_rap.txt", 'w') as file:
    file.writelines(concatenated_rap_lyrics)



#%% TODO

# remove english / remove

# English : Famous Dex / Bun B / Rilès / ejmatt
# Calboy / Menelik / Muse / Pihpoh / the driver era  / Brand Nubian / Jay Rock
# Admiral T / Farid Bang / Passion Pit / Flynt Flossy / Carl Brave x Franco126
# Taylor Swift / A Boogie wit da Hoodie / Piloophaz
# IAM qqs chansons / MHD / Mister Mex / Nazar / IamSu!
# Big Red Machine / Axiom / Fisto / André 3000 / Fefe Dobson / 6LACK

# Entre {}

# 1er couplet

# Paroles trop courtes (pas de paroles)