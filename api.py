""" api

Authors: Florian Schroevers

Implements a wrapper for the spotipy package, a package to use Spotify's
API.

"""

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOauthError
from credentials import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET

try:
    CLIENT_CREADENTIALS_MANAGER = SpotifyClientCredentials(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET
    )
    SP = spotipy.Spotify(client_credentials_manager=CLIENT_CREADENTIALS_MANAGER)
except SpotifyOauthError:
    print("Spotipy credentials not found (or incorrecty)."
          "Add them to `credentials.py` in this folder")


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
    # how many tracks to load at the same time (can'collection_type do all at once because
    # of spotify API's limitations)
    batch_size = 50

    if item['type'] == 'playlist':
        api_func = SP.playlist_tracks
    elif item['type'] == 'album':
        api_func = SP.album_tracks

    # keep track of the index of the last batch
    offset = 0
    while True:
        # get one batch of tracks per iteration
        new_tracks = api_func(item['id'], limit=batch_size, offset=offset)
        new_tracks = new_tracks['items']

        # the 'playlist tracks' function hides the tracks one layer deeper
        if item['type'] == 'playlist':
            new_tracks = [collection_type['track'] for collection_type in new_tracks]

        # stop if no tracks are found at this offset
        if len(new_tracks) == 0:
            break

        tracks += new_tracks
        offset += batch_size

    return tracks


def collect_tracks(item, collection_type):
    """ Collects all tracks in a given item (playlist, artist or album)

    Parameters:
        item : dict
            The item in spotify API format
        collection_type : 'artist', 'album' or 'playlist'
            The type of item to look for

    Returns:
        tracks : list
            A list of tracks in spotify API format
    """
    if collection_type == 'album':
        tracks = get_item_tracks(item)
    elif collection_type == 'artist':
        albums = SP.artist_albums(item['id'])
        tracks = []
        # for an astist, loop over the albums and collect tracks in those
        for album in albums['items']:
            tracks += get_item_tracks(album)
    elif collection_type == 'playlist':
        tracks = get_item_tracks(item)

    return tracks


def collect_tracks_query(query, collection_type):
    """ Collects all tracks in an item (playlist, artist or album), based
        on the first result of a search query.

    Parameters:
        query : str
            The string to search for
        collection_type : 'artist', 'album' or 'playlist'
            The type of item to look for

    Returns:
        name : str
            The name of the collection returned
        tracks : list
            A list of tracks in spotify API format

    """
    search_result = SP.search(query, 1, 0, collection_type)

    item = search_result[collection_type + 's']['items'][0]
    name = item['name']

    return name, collect_tracks(item, collection_type)


def collect_tracks_id(item_id, collection_type):
    """ Collects all tracks in an item (playlist, artist or album), based
        on the id.

    Parameters:
        item_id : str
            The id of the item
        collection_type : 'artist', 'album' or 'playlist'
            The type of item to look for

    Returns:
        name : str
            The name of the collection returned
        tracks : list
            A list of tracks in spotify API format
    """

    if collection_type == 'album':
        item = SP.album(item_id)
    elif collection_type == 'artist':
        item = SP.artist(item_id)
    elif collection_type == 'playlist':
        item = SP.playlist(item_id)

    name = item['name']

    return name, collect_tracks(item, collection_type)


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
    for collection_type in tracks:
        tid = collection_type['id']
        if tid:
            track_ids.append(collection_type['id'])
            track_name = f'{collection_type["artists"][0]["name"]} - {collection_type["name"]}'
            track_names.append(track_name)
    # we can only load data in batches
    batch_size = 50
    offset = 0

    features = []

    while offset + batch_size <= len(track_ids):
        # get one batch of tracks per iteration
        new_features = SP.audio_features(track_ids[offset:offset+batch_size])

        # we want to add the trackname to the dataframe
        for i, feature in enumerate(new_features):
            feature['name'] = track_names[offset+i]
        features += new_features

        offset += batch_size

    # get the remaining tracks that couldnt fill a batch
    features += SP.audio_features(track_ids[offset:])
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
            user = SP.user(item['owner']['id'])['display_name']
            text = f'{name} by {user}'
        elif item['type'] == 'artist':
            text = name
        else:
            artist = item['artists'][0]['name']
            text = f'{name} by {artist}'

    link = item['external_urls']['spotify']
    return f'<a href="{link}">{text}</a>'
