""" ga

Authors: Florian Schroevers

Implements a genetic algorithm to find an optimal path. Paths are assigned
a fitness score based on length of the path, overall distance travelled, and
angles between transitions.

TODO: clustering

"""

import numpy as np
from scipy.spatial.distance import cosine, euclidean
import matplotlib.pyplot as plt

import visualization


def initialize_population(tracks, population_size):
    """ Initializes a population of a given size. One element of the
        population, or an individual, represents a path through the tracks.
        The path is given by the indices of the tracks.

    Paramters:
        tracks : np.array
            An NxM array of tracks, containing N tracks with M features each.
        population_size : int
            The size of the population

    Returns:
        population : list
            A list of 'individuals', each is a list of indices of the tracks
            array.
    """
    population = []

    for _ in range(population_size):
        # n_tracks = np.random.randint(5, len(tracks))


        # choose a random sample from the indices of the tracks
        individual = np.random.choice(
            list(range(len(tracks))),
            len(tracks),
            replace=False
        )

        population.append(list(individual))

    return population


def get_fitness(tracks, individual):
    """ Calculates the fitness of an individual based on the total length
        of the path, the number of tracks it contains, and the total angular
        change.

    Paramters:
        tracks : np.array
            An NxM array of tracks, containing N tracks with M features each.
        individual : list
            An individual, or a list of indices of the tracks array.

    Returns:
        fitness
    """
    cossim_cost = 0
    distance_cost = 0

    for i in range(1, len(individual) - 1):
        previous_track = tracks[individual[i - 1]]
        current_track = tracks[individual[i]]
        next_track = tracks[individual[i + 1]]

        cossim = cosine(
            current_track - previous_track,
            next_track - current_track
        )

        cossim_cost += cossim

        distance = euclidean(previous_track, current_track)
        distance_cost += distance

    # add distance of last transition
    distance_cost += euclidean(tracks[individual[-2]], tracks[individual[-1]])

    # longer paths get a discount
    # path_length_cost = -(len(individual)/len(tracks))

    # fitness is negative cost
    return -(distance_cost**0.5)


def select(tracks, population, death_rate=0.75):
    fitnessess = []

    for individual in population:
        fitnessess.append(get_fitness(tracks, individual))

    print(f"Mean fitness: {np.mean(fitnessess)}", end='\r')

    population_fitness = sorted(
        zip(population, fitnessess),
        key=lambda x: x[1],
        reverse=True
    )

    population = []

    for pop, fit in population_fitness:
        population.append(pop)
        fitnessess.append(fit)

    n_surviving = int(death_rate * len(population))

    return population[:n_surviving], fitnessess


def mutate(population, max_gene, mutation_prob=0.005):
    for individual in population:
        for gene_i, _ in enumerate(individual):
            if np.random.rand() < mutation_prob:
                other_gene_i = np.random.randint(len(individual))
                tmp = individual[other_gene_i]
                individual[other_gene_i] = individual[gene_i]
                individual[gene_i] = tmp

        # if np.random.random() < mutation_prob:
        #   add_gene = np.random.randint(max_gene)
        #   if add_gene not in individual:
        #       add_i = np.random.randint(len(individual) + 1)
        #       individual = individual[:add_i] + [add_gene] + individual[add_i:]

        #   remove_i = np.random.randint(len(individual))
        #   individual = individual[:remove_i] + individual[remove_i + 1:]

    return population

def copulate(parent1, parent2):
    startend = np.random.choice(list(range(len(parent1))), 4, replace=False)

    start, end = min(startend), max(startend)
    interval = parent1[start:end]

    child = []
    i = 0

    for i, k in enumerate(parent2):
        if i == start:
            child += interval

        if k not in interval:
            child.append(k)

    if i < start:
        child += interval

    assert len(set(child)) == len(child)
    return child

def breed(population, birth_rate=2):
    n_children = int(len(population) * (birth_rate - 1))

    # TODO: make sure every individual breeds (maybe?)

    for _ in range(n_children):
        i, j = np.random.choice(list(range(len(population))), 2, replace=False)
        population.append(copulate(population[i], population[j]))

    return population


def genetic_algorithm(tracks, n_generations=3000, population_size=100, plot=False):
    population = initialize_population(tracks, population_size)

    best_fitness = -float("inf")

    mean_fitness_per_generation = []
    min_fitness_per_generation = []
    max_fitness_per_generation = []

    for generation in range(n_generations):
        print(f"Generation: {generation + 1}", end=', ')
        population = mutate(population, len(tracks))
        population, fitness = select(tracks, population)
        population = breed(population)
        population = population[:population_size]

        if fitness[0] > best_fitness:
            best_individual = population[0]
            best_fitness = fitness[0]

        mean_fitness_per_generation.append(np.mean(fitness))
        min_fitness_per_generation.append(np.min(fitness))
        max_fitness_per_generation.append(np.max(fitness))

        if plot:
            visualization.plot_path(
                tracks,
                population[0],
                fitness[0],
                mode='none'
            )

    plt.plot(mean_fitness_per_generation)
    plt.plot(min_fitness_per_generation)
    plt.plot(max_fitness_per_generation)
    plt.show()

    return best_individual, best_fitness


if __name__ == "__main__":
    genetic_algorithm(np.random.rand((5, 2)))
