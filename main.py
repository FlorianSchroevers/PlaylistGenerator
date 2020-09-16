""" main

Authors: Florian Schroevers

"""
import sys

import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cosine

import api
import visualization

def playlist_transition_matrix(track_features):
    """ Returns a matrix of all possible `transitions`; vectors that are
        simply the difference of two track vectors.

    Parameters:
        track_features : np.array
            An N x M array where N is the number of tracks and M is the
            number of features

    Returns:
        transition_matrix : np.array
            N x N x M array of all possible transitions

    """
    n_transitions = track_features.shape[0]
    n_features = track_features.shape[1]

    # intialize empty return array of shape N x N x M
    transition_matrix = np.empty((n_transitions, n_transitions, n_features))

    # loop pairs of tracks
    for i, track_vec in enumerate(track_features):
        for j, other_track_vec in enumerate(track_features):
            # transitions to self are 0
            if i == j:
                transition_matrix[i, j, :] = 0
            else:
                transition_matrix[i, j, :] = other_track_vec - track_vec

    return transition_matrix


def playlist_transition_score_matrix(track_features):
    """ Associates a score to all possible pairs of transitions
    (see function `playlist_transition_matrix`). A pair of transitions
    will consist of three tracks, such that there are two transitions
    (track a -> track b, track b -> track c). Each pair of transitions
    will get a score based on the cosine similarity between two transitions.
    This will ensure the playlist will more or less progress in the same
    direction.

    Parameters:
        track_features : np.array
            An N x M array where N is the number of tracks and M is the
            number of features

    Returns:
        distance_matrix : np.array
            N x N array of euclidean distances between tracks
        transition_followup_matrix : np.array
            N x N x N array of scores associated to transition pairs

    """

    transition_matrix = playlist_transition_matrix(track_features)

    # transition matrix holds vectors 'between' tracks, the length of which
    # is the euclidean distance
    distance_matrix = np.linalg.norm(transition_matrix, axis=-1)

    n_tracks = track_features.shape[0]
    transtition_followup_matrix = np.empty((n_tracks, n_tracks, n_tracks))

    # TODO: optimize this
    # loop over triplets of tracks
    for i in range(n_tracks):
        for j in range(n_tracks):
            # don't check transitions to itself
            if i == j:
                transtition_followup_matrix[i, j, :] = 0
            else:
                for k in range(n_tracks):
                    # don't check transitions to itself
                    if j == k:
                        transtition_followup_matrix[:, j, k] = 0
                    # make sure the second transition does not go to
                    # the initial track
                    elif i == k:
                        transtition_followup_matrix[i, :, k] = 0
                    else:
                        # get the two transitions and calculate the cosine
                        # similarity, and store in the output array
                        transition1 = (transition_matrix[i, j])
                        transition2 = (transition_matrix[j, k])
                        transtition_followup_matrix[i, j, k] = \
                            cosine(transition1, transition2) - 1

    return distance_matrix, transtition_followup_matrix


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

    tracknames = list(playlist['name'])
    track_features = playlist[['danceability', 'energy', 'loudness',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
        'valence']]

    distance_matrix, transtition_followup_matrix = \
        playlist_transition_score_matrix(track_features.values)


    visualization.scatter(
        track_features, 
        distance_matrix, 
        transtition_followup_matrix,
        dim=3
    )

    return tracklist

def main(args):
    name, tracks = api.collect_tracks_query('psytrance', 'playlist')
    playlist = api.get_tracklist_features(tracks)
    recommendation = make_recommendation(playlist)

    print(recommendation)


if __name__ == '__main__':
    #argparser
    args = sys.argv
    main(*args)
