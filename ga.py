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
    """ Uses natural selection to slim down a given population. For each
        individual a fitness score will be calculated. The individuals
        with a lower fitness (the fraction of which can be specified) die.

    Paramters:
        tracks : np.array
            An NxM array of tracks, containing N tracks with M features each.
        population : list
            A list containing individuals.
        death_rate : float (optional, default: 0.75)
            The fraction of individuals to die during selection

    Returns:
        population : list
            The surviving population
        fitnessess : list
            The fitnessess of the surviving population, matched with
            'population' by index. The fitnessess are of the entire population
            before selection.
    """
    fitnessess = []

    # get list of fitnessess
    for individual in population:
        fitnessess.append(get_fitness(tracks, individual))

    print(f"Mean fitness: {np.mean(fitnessess)}", end='\r')

    # get a sorted list of a list of tuples containing indivuiduals and their fitness
    population_fitness = sorted(
        zip(population, fitnessess),
        key=lambda x: x[1],
        reverse=True
    )

    population = []

    # unzip the list of tuples
    for pop, fit in population_fitness:
        population.append(pop)
        fitnessess.append(fit)

    # only keep the top scoring individuals
    n_surviving = int(death_rate * len(population))
    population = population[:n_surviving]

    return population, fitnessess


def mutate(population, mutation_prob=0.005):
    """ Do random mutation of the genes in a given population based on some
        probability.

    Paramters:
        population : list
            A list containing individuals.
        mutation_prob : float (optional, default: 0.005)
            The probability of a mutation occuring in a gene.

    Returns:
        population : list
            The mutated population
    """
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
    """ Combine two individuals into an offspring.

    Paramters:
        parent1 : list
            The gene of the first parent.
        parent2 : list
            The gene of the second parent.

    Returns:
        child : list
            The gene of the offspring.
    """

    # randomly choose the bounding indices of a section of the DNA from parent1
    startend = np.random.choice(list(range(len(parent1))), 4, replace=False)

    start, end = min(startend), max(startend)
    interval = parent1[start:end]

    child = []
    i = 0

    # add all genes from parent 2, that are not in the above interval, and add
    # them in order to the child, if the index is in the interval, add the gene
    # from parent one.
    for i, k in enumerate(parent2):
        if i == start:
            child += interval

        if k not in interval:
            child.append(k)

    if i < start:
        child += interval

    # make sure the child's dna contains unique genes
    assert len(set(child)) == len(child)
    return child

def breed(population, birth_rate=2.):
    """ Perform a batch of copulation between the individuals in a given
        population by combining individuals.

    Paramters:
        population : list
            A list containing individuals.
        birth_rate : float (optional, default: 2.)
            The number by which to multiply the number of individuals.
            (2 means the population will be doubled)

    Returns:
        population : list
            A list containing individuals including all new offspring.
    """
    n_children = int(len(population) * (birth_rate - 1))

    # IDEA: make sure every individual breeds (maybe?)

    for _ in range(n_children):
        # choose two random individuals from the population and breed them
        i, j = np.random.choice(list(range(len(population))), 2, replace=False)
        population.append(copulate(population[i], population[j]))

    return population


def genetic_algorithm(tracks, n_generations=3000, population_size=100, plot=False):
    """ Perform a Genetic Algorithm (GA) on an array of tracks to find the best
        fitting path (order for the tracks).

    Paramters:
        tracks : np.ndarray
            An array of points, each point represents a track.
        n_generations : int (optional, default: 3000)
            The number of generations
        population_size : int (optional, default: 100)
            The size of the population
        plot : bool (optional, default: False)
            Whether to plot the best path of each generation.

    Returns:
        best_individual : list
            The best path found by the algorithm.
        best_fitness : float
            The fitness of the best path found by the algorithm.
    """
    population = initialize_population(tracks, population_size)

    best_fitness = -float("inf")

    mean_fitness_per_generation = []
    min_fitness_per_generation = []
    max_fitness_per_generation = []

    for generation in range(n_generations):
        print(f"Generation: {generation + 1}", end=', ')
        population = mutate(population)
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
    genetic_algorithm(np.random.rand(5, 2))
