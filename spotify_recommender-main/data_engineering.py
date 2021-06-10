#!/usr/bin/env python
# coding: utf-8

# In[27]:


#Импорт либ
import time
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

import random
from functools import reduce
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import oauth2


# In[28]:


#Все данные со споти дева
cid = '02d33d68c8724a3fb5209f420a3e4d1a'
secret = '7e1ecf62485a4fec84700590a1453e9f'
redirect_uri='http://localhost:7777/callback'
username = '9xg1dh4sogilprh9kw0cdw3wi'


# In[29]:


#Авторизация с данными в споти и необходимыми скоупами
scope = 'user-top-read playlist-modify-private playlist-modify-public'
token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret, redirect_uri=redirect_uri)

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)


# In[30]:


# Эту функцию скопипастил
def fetch_audio_features(sp, df):
    playlist = df[['track_id','track_name']] 
    index = 0
    audio_features = []
    
    # Make the API request
    while index < playlist.shape[0]:
        audio_features += sp.audio_features(playlist.iloc[index:index + 50, 0])
        index += 50
    
    # Create an empty list to feed in different charactieritcs of the tracks
    features_list = []
    #Create keys-values of empty lists inside nested dictionary for album
    for features in audio_features:
        features_list.append([features['danceability'],
                              features['acousticness'],
                              features['energy'], 
                              features['tempo'],
                              features['instrumentalness'], 
                              features['loudness'],
                              features['liveness'],
                              features['duration_ms'],
                              features['key'],
                              features['valence'],
                              features['speechiness'],
                              features['mode']
                             ])
    
    df_audio_features = pd.DataFrame(features_list, columns=['danceability', 'acousticness', 'energy','tempo', 
                                                             'instrumentalness', 'loudness', 'liveness','duration_ms', 'key',
                                                             'valence', 'speechiness', 'mode'])
    
    # Запихиваем это всё в датафрэйм
    df_playlist_audio_features = pd.concat([playlist, df_audio_features], axis=1)
    df_playlist_audio_features.set_index('track_name', inplace=True, drop=True)
    return df_playlist_audio_features


# ###Получение всех песен
# 

# Получение всех песен с спотифая, занимает очень много времени и я уже это сделал так что закомментил

# In[31]:


# # Getting playlist IDs from each of Spotify's playlists
# playlists = sp.user_playlists('spotify')
# spotify_playlist_ids = []
# while playlists:
#     for i, playlist in enumerate(playlists['items']):
#         spotify_playlist_ids.append(playlist['uri'][-22:])
#     if playlists['next']:
#         playlists = sp.next(playlists)
#     else:
#         playlists = None
# spotify_playlist_ids[:20]


# In[32]:


# len(spotify_playlist_ids)


# ### Получение самых прослушиваемых песен с аккаунта.

# In[33]:



def getTrackIDs(playlist_id):
    playlist = sp.user_playlist('spotify', playlist_id)
    for item in playlist['tracks']['items'][:50]:
        track = item['track']
        ids.append(track['id'])
    return


# In[34]:



def getTrackFeatures(track_id):
  meta = sp.track(track_id)
  features = sp.audio_features(track_id)

  # meta
  track_id = track_id
  name = meta['name']
  album = meta['album']['name']
  artist = meta['album']['artists'][0]['name']
  release_date = meta['album']['release_date']
  length = meta['duration_ms']
  popularity = meta['popularity']

  # features
  acousticness = features[0]['acousticness']
  danceability = features[0]['danceability']
  energy = features[0]['energy']
  instrumentalness = features[0]['instrumentalness']
  liveness = features[0]['liveness']
  loudness = features[0]['loudness']
  speechiness = features[0]['speechiness']
  tempo = features[0]['tempo']
  time_signature = features[0]['time_signature']

  track = [track_id, name, album, artist, release_date, length, popularity, danceability, acousticness, energy, instrumentalness, liveness, loudness, speechiness, tempo, time_signature]
  return track


# In[35]:


#Датасет
df = pd.read_csv('data/playlist_songs.csv')



# In[36]:


# Убираем лишние колонки
df = df.drop(columns=['name', 'album', 'artist', 'release_date'])



# In[37]:


# Убираем повторяющиеся
df = df.drop_duplicates(subset=['track_id'])



# In[ ]:





# ## Getting user's favorite tracks

# In[38]:


# Топ 50 песен пользователя
results = sp.current_user_top_tracks(limit=1000, offset=0,time_range='short_term')


# In[39]:


# Конвертируем в формат датасета
track_name = []
track_id = []
artist = []
album = []
duration = []
popularity = []
for i, items in enumerate(results['items']):
        track_name.append(items['name'])
        track_id.append(items['id'])
        artist.append(items["artists"][0]["name"])
        duration.append(items["duration_ms"])
        album.append(items["album"]["name"])
        popularity.append(items["popularity"])

# Create the final df   
df_favourite = pd.DataFrame({ "track_name": track_name, 
                             "album": album, 
                             "track_id": track_id,
                             "artist": artist, 
                             "duration": duration, 
                             "popularity": popularity})




# In[40]:

# Собираем все параметры любимых песен
fav_tracks = []
for track in df_favourite['track_id']:
    try:
        track = getTrackFeatures(track)
        fav_tracks.append(track)
    except:
        pass



# In[41]:


# Создаем датасет
df_fav = pd.DataFrame(fav_tracks, columns = ['track_id', 'name', 'album', 'artist', 'release_date', 'length', 'popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'time_signature'])
df_fav.head()


# In[42]:


# Опять сносим колонки
df_fav = df_fav.drop(columns=['name', 'album', 'artist', 'release_date'])
df_fav.head()


# In[43]:


#Убираем дупликаты
df_fav['track_id'].value_counts()


# In[44]:


# Создаём колонку "любимые" для классификации
df_fav['favorite'] = 1
df['favorite'] = 0 


# In[45]:


# Сравниваем колонки двух датасетов



# ## Подготовка к созданию модели

# In[46]:




# In[47]:


# Объединяем датасет всех песен с любимыми
combined = pd.concat([df, df_fav])



# In[48]:


combined.favorite.value_counts()


# In[49]:


# Датафрэйм с любимыми песнями
df_fav = combined.loc[combined['favorite'] == 1]
df_fav.head()


# In[50]:


# Убираем любимые песни с плэйлиста
df = combined.loc[combined['favorite'] != 1]



# In[51]:





# In[52]:


# Сохраняем в csv для создания модели
df.to_csv('encoded_playlist_songs.csv', index=False)
df_fav.to_csv('favorite_songs.csv', index=False)


# In[ ]:




