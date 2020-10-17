""" main

Authors: Florian Schroevers

TODO: add main algorithm and improve visualization

"""
import sys

import numpy as np

import api
import visualization
import search
import ga


def make_recommendation(playlist):
    """ Makes a recomendation to add to playlist based on a set of tracks

    Paramters:
        track_features : pandas DataFrame
            The tracks to base the recomendation from. This should be the
            output of `api.get_tracklist_features()`.

    Returns:
        recommendations : pandas DataFrame
            The recommendations based on the input.

    """
    tracklist = []

    # tracknames = list(playlist['name'])

    track_features = playlist[['danceability', 'energy']]
                               # 'speechiness', 'acousticness',
                               # 'instrumentalness', 'liveness', 'valence']]

    track_features_matrix = track_features.values

    path, fitness = ga.genetic_algorithm(track_features_matrix)

    visualization.plot_path(
        track_features,
        path,
        fitness,
        mode="none",
        keep=True
    )

    return tracklist


def main(args):
    name, tracks = api.collect_tracks_query('psytrance', 'playlist')
    playlist = api.get_tracklist_features(tracks)
    recommendation = make_recommendation(playlist)

    print(recommendation)


if __name__ == '__main__':
    # argparser
    args = sys.argv
    main(*args)
