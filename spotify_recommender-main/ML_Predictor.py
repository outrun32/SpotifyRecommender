#!/usr/bin/env python
# coding: utf-8


import json
import os
import random
# Импорт либ
import time
from functools import reduce

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
import spotipy
import spotipy.util as util
from imblearn.over_sampling import SMOTE
from spotipy import oauth2
from spotipy.oauth2 import SpotifyClientCredentials

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics 
from sklearn.metrics import f1_score


from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class SpotifyData:
    def __init__(self, username):
        # Все данные со споти дева
        self.cid = '02d33d68c8724a3fb5209f420a3e4d1a'
        self.secret = '7e1ecf62485a4fec84700590a1453e9f'
        self.redirect_uri = 'http://localhost:7777/callback'
        self.username = username

        # Авторизация с данными в споти и необходимыми скоупами
        self.scope = 'user-top-read playlist-modify-private playlist-modify-public'
        self.token = util.prompt_for_user_token(username, self.scope, client_id=self.cid, client_secret=self.secret,
                                                redirect_uri=self.redirect_uri)
        if self.token:
            self.sp = spotipy.Spotify(auth=self.token)

        # Эту функцию скопипастил
    def fetch_audio_features(self, sp, df):
        playlist = df[['track_id', 'track_name']]
        index = 0
        audio_features = []

        # Make the API request
        while index < playlist.shape[0]:
            audio_features += sp.audio_features(playlist.iloc[index:index + 50, 0])
            index += 50

        # Create an empty list to feed in different charactieritcs of the tracks
        features_list = []
        # Create keys-values of empty lists inside nested dictionary for album
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

        df_audio_features = pd.DataFrame(features_list, columns=['danceability', 'acousticness', 'energy', 'tempo',
                                                             'instrumentalness', 'loudness', 'liveness', 'duration_ms',
                                                             'key',
                                                             'valence', 'speechiness', 'mode'])

        # Запихиваем это всё в датафрэйм
        df_playlist_audio_features = pd.concat([playlist, df_audio_features], axis=1)
        df_playlist_audio_features.set_index('track_name', inplace=True, drop=True)
        return df_playlist_audio_features



    def getTrackFeatures(self, track_id):
        meta = self.sp.track(track_id)
        features = self.sp.audio_features(track_id)

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

        track = [track_id, name, album, artist, release_date, length, popularity, danceability, acousticness, energy,
             instrumentalness, liveness, loudness, speechiness, tempo, time_signature]
        return track

    def final_prep(self):
        df = pd.read_csv('data/playlist_songs.csv')

        # Убираем лишние колонки
        df = df.drop(columns=['name', 'album', 'artist', 'release_date'])

        # Убираем повторяющиеся
        df = df.drop_duplicates(subset=['track_id'])

        # Топ 50 песен пользователя
        results = self.sp.current_user_top_tracks(limit=1000, offset=0, time_range='short_term')

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
        df_favourite = pd.DataFrame({"track_name": track_name,
                             "album": album,
                             "track_id": track_id,
                             "artist": artist,
                             "duration": duration,
                             "popularity": popularity})

        # Собираем все параметры любимых песен
        fav_tracks = []
        for track in df_favourite['track_id']:
            try:
                track = self.getTrackFeatures(track)
                fav_tracks.append(track)
            except:
                pass

        # Создаем датасет
        df_fav = pd.DataFrame(fav_tracks,
                      columns=['track_id', 'name', 'album', 'artist', 'release_date', 'length', 'popularity',
                               'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness',
                               'speechiness', 'tempo', 'time_signature'])

        # Опять сносим колонки
        df_fav = df_fav.drop(columns=['name', 'album', 'artist', 'release_date'])

        # Убираем дупликаты
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

        # Убираем любимые песни с плэйлиста
        df = combined.loc[combined['favorite'] != 1]

         # Сохраняем в csv для создания модели
        df.to_csv('encoded_playlist_songs.csv', index=False)
        #df_fav.to_csv('favorite_songs.csv', index=False)
        return df, df_fav


class ModelTraining:
    def __init__(self, df, df_fav):
        self.df = df
        self.df_fav = df_fav
    
    def prepare_data(self):
        self.df = pd.concat([self.df, self.df_fav], axis=0)

    def split_data(self):
        self.prepare_data()
        # Перемешиваем
        shuffle_df = self.df.sample(frac=1)

        # Обьявлем размер тестового (80% как обычно) 
        train_size = int(0.8 * len(self.df))

        # Разделяем на тестовый и тренировочный
        train_set = shuffle_df[:train_size]
        test_set = shuffle_df[train_size:]
        X = train_set.drop(columns=['favorite', 'track_id'])
        y = train_set.favorite


        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X, y)

        X_test = test_set.drop(columns=['favorite', 'track_id'])
        y_test = test_set['favorite']    

        return X_train, y_train, X_test, y_test

    def train_model(self):
        X_train, y_train, X_test, y_test = self.split_data()

        parameters = {
        'max_depth':[3, 4, 5, 6, 10, 15,20,30],
        }
        dtc = Pipeline([('CV',GridSearchCV(DecisionTreeClassifier(), parameters, cv = 5))])
        dtc.fit(X_train, y_train)
        best_depth = dtc.named_steps['CV'].best_params_["max_depth"]
        dt = DecisionTreeClassifier(max_depth=best_depth).fit(X_train, y_train)
        pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=30))
        pipe.fit(X_train, y_train)  # apply scaling on training data
        Pipeline(steps=[('standardscaler', StandardScaler()),
                ('dt', DecisionTreeClassifier(max_depth=30))])
        return pipe

    def make_predictions(self):
        pipe = self.train_model()
        df = pd.read_csv('data/encoded_playlist_songs.csv')
        prob_preds = pipe.predict_proba(df.drop(['favorite','track_id'], axis=1))
        threshold = 0.30
        preds = [1 if prob_preds[i][1]> threshold else 0 for i in range(len(prob_preds))]
        df['prediction'] = preds
        return df


class Spotify:
    def __init__(self, sp, username, df):
        self.sp = sp
        self.create_playlist(sp, username, 'Это предсказано нейросетью 2', 'Спасибо API Spotify')
        playlist_id = self.fetch_playlists(sp,username)['id'][0]
        list_track = df.loc[df['prediction']  == 1]['track_id']
        self.enrich_playlist(sp, username, playlist_id, list_track)
    def create_playlist(self, sp, username, playlist_name, playlist_description):
        self.playlists = sp.user_playlist_create(username, playlist_name, description = playlist_description)
    def fetch_playlists(self, sp, username):
        
        id = []
        name = []
        num_tracks = []
    
        # Make the API request
        playlists = sp.user_playlists(username)
        for playlist in playlists['items']:
            id.append(playlist['id'])
            name.append(playlist['name'])
            num_tracks.append(playlist['tracks']['total'])

        # Create the final df   
        df_playlists = pd.DataFrame({"id":id, "name": name, "#tracks": num_tracks})
        return df_playlists
    def enrich_playlist(self, sp, username, playlist_id, playlist_tracks):
        index = 0
        results = []
        
        while index < len(playlist_tracks):
            results += sp.user_playlist_add_tracks(username, playlist_id, tracks = playlist_tracks[index:index + 50])
            index += 50

def set_results(username):
    spD = SpotifyData(username)
    df, df_fav = spD.final_prep()
    md = ModelTraining(df, df_fav)
    predictions = md.make_predictions()
    Spotify(spD.sp, spD.username, predictions)