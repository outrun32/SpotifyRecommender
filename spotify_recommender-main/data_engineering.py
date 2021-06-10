#!/usr/bin/env python
# coding: utf-8


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



class SpotifyData:

#Все данные со споти дева
    def __init__(self, username):
        self.cid = '02d33d68c8724a3fb5209f420a3e4d1a'
        self.secret = '7e1ecf62485a4fec84700590a1453e9f'
        self.redirect_uri='http://localhost:7777/callback'
        self.username = username


        scope = 'user-top-read playlist-modify-private playlist-modify-public'
        self.token = util.prompt_for_user_token(username, scope, client_id=self.cid, client_secret=self.secret, redirect_uri=self.redirect_uri)

        if self.token:
            self.sp = spotipy.Spotify(auth=self.token)

#Авторизация с данными в споти и необходимыми скоупами


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

#Датасет
df = pd.read_csv('data/playlist_songs.csv')

# Убираем лишние колонки
df = df.drop(columns=['name', 'album', 'artist', 'release_date'])

# Убираем повторяющиеся
df = df.drop_duplicates(subset=['track_id'])

# Топ 50 песен пользователя
results = sp.current_user_top_tracks(limit=1000, offset=0,time_range='short_term')

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

# Собираем все параметры любимых песен
fav_tracks = []
for track in df_favourite['track_id']:
    try:
        track = getTrackFeatures(track)
        fav_tracks.append(track)
    except:
        pass

# Создаем датасет
df_fav = pd.DataFrame(fav_tracks, columns = ['track_id', 'name', 'album', 'artist', 'release_date', 'length', 'popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'time_signature'])

# Опять сносим колонки
df_fav = df_fav.drop(columns=['name', 'album', 'artist', 'release_date'])

#Убираем дупликаты
df_fav['track_id'].value_counts()

# Создаём колонку "любимые" для классификации
df_fav['favorite'] = 1
df['favorite'] = 0 

# Сравниваем колонки двух датасетов
# ## Подготовка к созданию модели

# Объединяем датасет всех песен с любимыми
combined = pd.concat([df, df_fav])






# Датафрэйм с любимыми песнями
df_fav = combined.loc[combined['favorite'] == 1]
df_fav.head()





# Убираем любимые песни с плэйлиста
df = combined.loc[combined['favorite'] != 1]


# Сохраняем в csv для создания модели
df.to_csv('encoded_playlist_songs.csv', index=False)
df_fav.to_csv('favorite_songs.csv', index=False)






