""" api

Authors: Florian Schroevers

Implements a wrapper for the spotipy package, a package to use Spotify's
API.

"""

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from credentials import *

try:
    client_credentials_manager = SpotifyClientCredentials(
        client_id = SPOTIPY_CLIENT_ID,
        client_secret = SPOTIPY_CLIENT_SECRET
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
except SpotifyOauthError:
    print("Spotipy credentials not found (or incorrecty). Add them to `credentials.py` in this folder")

def get_item_tracks(item):
    """ Returns a list of tracks from a given album or playlist

    Parameters:
        album : dict or str
            An album or playlist as returned by spotify's API

    Returns:
        tracks : list
            A list of tracks (in spotify format)

    """
    tracks = []
    # how many tracks to load at the same time (can't do all at once because
    # of spotify API's limitations)
    batch_size = 50

    if item['type'] == 'playlist':
        api_func = sp.playlist_tracks
    elif item['type'] == 'album':
        api_func = sp.album_tracks

    # keep track of the index of the last batch
    offset = 0
    while True:
        # get one batch of tracks per iteration
        new_tracks = api_func(item['id'], limit=batch_size, offset=offset)
        new_tracks = new_tracks['items']

        # the 'playlist tracks' function hides the tracks one layer deeper
        if item['type'] == 'playlist':
            new_tracks = [t['track'] for t in new_tracks]

        # stop if no tracks are found at this offset
        if not len(new_tracks):
            break

        tracks += new_tracks
        offset += batch_size

    return tracks


def collect_tracks(item, t):
    """ Collects all tracks in a given item (playlist, artist or album)

    Parameters:
        item : dict
            The item in spotify API format
        t : 'artist', 'album' or 'playlist'
            The type of item to look for

    Returns:
        tracks : list
            A list of tracks in spotify API format
    """
    if t == 'album':
        tracks = get_item_tracks(item)
    elif t == 'artist':
        albums = sp.artist_albums(item['id'])
        tracks = []
        # for an astist, loop over the albums and collect tracks in those
        for album in albums['items']:
            tracks += get_item_tracks(album)
    elif t == 'playlist':
        tracks = get_item_tracks(item)

    return tracks


def collect_tracks_query(query, t):
    """ Collects all tracks in an item (playlist, artist or album), based
        on the first result of a search query.

    Parameters:
        query : str
            The string to search for
        t : 'artist', 'album' or 'playlist'
            The type of item to look for

    Returns:
        name : str
            The name of the collection returned
        tracks : list
            A list of tracks in spotify API format

    """
    search_result = sp.search(query, 1, 0, t)

    item = search_result[t + 's']['items'][0]
    name = item['name']

    return name, collect_tracks(item, t)


def collect_tracks_id(item_id, t):
    """ Collects all tracks in an item (playlist, artist or album), based
        on the id.

    Parameters:
        item_id : str
            The id of the item
        t : 'artist', 'album' or 'playlist'
            The type of item to look for

    Returns:
        name : str
            The name of the collection returned
        tracks : list
            A list of tracks in spotify API format
    """

    if t == 'album':
        item = sp.album(item_id)
    elif t == 'artist':
        item = sp.artist(item_id)
    elif t == 'playlist':
        item = sp.playlist(item_id)

    name = item['name']

    return name, collect_tracks(item, t)


def get_tracklist_features(tracks):
    """ Given a list of spotify tracks, get all spotify features of
        those tracks and put them in a pandas DataFrame

    Parameters:
        tracks : list
            a list of tracks in spotify format

    Returns:
        features : pandas DataFrame
            the dataframe of features
    """

    # first we construct a list of all track ids and tracknames
    track_ids = []
    track_names = []
    for t in tracks:
        tid = t['id']
        if tid:
            track_ids.append(t['id'])
            track_names.append(f'{t["artists"][0]["name"]} - {t["name"]}')
    # we can only load data in batches
    batch_size = 50
    offset = 0

    features = []

    while offset + batch_size <= len(track_ids):
        # get one batch of tracks per iteration
        nf = sp.audio_features(track_ids[offset:offset+batch_size])

        # we want to add the trackname to the dataframe
        for i, f in enumerate(nf):
            f['name'] = track_names[offset+i]
        features += nf

        offset += batch_size

    # get the remaining tracks that couldnt fill a batch
    features += sp.audio_features(track_ids[offset:])
    return pd.DataFrame(features)


def wrap_spotify_link(item, text=''):
    """ Makes a HTML link out of a given spotify item (artist, album,
        playlist or track). Has default text for each type of item
        ({name} by {creator}).

    Parameters:
        item : dict
            A spotify item
        text : str (optional, default: "")
            An alternative text.

    Returns:
        link : str
            An HTML link refering to the given item
    """

    # generate default text if no text has been given
    if not text:
        name = item['name']
        if item['type'] == 'playlist':
            user = sp.user(item['owner']['id'])['display_name']
            text = f'{name} by {user}'
        elif item['type'] == 'artist':
            text = name
        else:
            artist = item['artists'][0]['name']
            text = f'{name} by {artist}'

    link = item['external_urls']['spotify']
    return f'<a href="{link}">{text}</a>'
