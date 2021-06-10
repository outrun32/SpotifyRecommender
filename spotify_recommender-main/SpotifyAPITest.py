import spotipy
from spotipy.oauth2 import SpotifyOAuth
from pprint import pprint

cid = '758751bbdb8c458db1fb14d48e994e7a'
secret = '51f0a8678a4a440c97d7636ac29dee9a'
redirect_uri = 'http://localhost:7777/callback'
username = 'outrun32'

scope = 'user-top-read playlist-modify-private playlist-modify-public'
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cid, client_secret=secret, redirect_uri=redirect_uri,
                                               username=username, scope=scope))
