import api.azapi
import numpy as np
import time

# songs = []

with open('ok.txt', 'r') as f:
    songs = f.read().splitlines()
    # [line.split(',') for line in f]
    # newsong = songs.split(',')
# training_inputs  = np.array(songs, dtype=str)


# print songs
def test():
    # newsong = [i[0] for i in songs]

    for i in range (len(songs)):
        # songs[i]
    	artist = 'eminem'
        print artist
        title = songs[i]
        print title
        api.azapi.generating(artist, title, save=True)

        time.sleep(3)

        #raw_input("Insert artist: ")

    # = raw_input("Insert title: ")

	# api.azapi.generating(artist, title, save=True)


if __name__ == '__main__':
    test()


"https://genius.com/artists/Eminem"
