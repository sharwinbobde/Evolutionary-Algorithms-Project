import random
from math import floor
import matplotlib.pyplot as plt

from pyeasyga import pyeasyga

from utils.graph_manager import GraphManager
from fastga import FastMutationOperator

MAX_EVALUATIONS = 100000
evaluations = 0


def get_relative_neighbours(node_count, edges):
    neighbours = [0] * node_count
    for e in edges:
        neighbours[e[0] - 1] += 1
        neighbours[e[1] - 1] += 1
    max_neighbours = max(neighbours)
    relative_neighbours = [x / max_neighbours for x in neighbours]
    return relative_neighbours


def get_mask_percent(relative_neighbours):
    return [pow(x, 2.0) for x in relative_neighbours]


def get_mask(mask_percent):
    return [x > random.uniform(0, 1) for x in mask_percent]


def fitness(individual, data):
    global evaluations
    evaluations += 1

    fitness_score = data[0].evaluate_individual(individual)
    return fitness_score


def create_individual(data):
    parameter_count = data[0].n
    return [random.randint(0, 1) for _ in range(parameter_count)]


def mutate_individual(individual):
    mutated_individual = fast_mutation_operator.mutate(individual, inplace=False)
    for i in range(0, len(individual)):
        if mask[i]:
            individual[i] = mutated_individual[i]


def n_req_analysis(graph_manager, beta):
    n = 1
    solved_consistently = False
    while not solved_consistently:
        n *= 2
        # print("Solving for n =", n, end=' ')
        generations = MAX_EVALUATIONS / n
        if can_solve_10_times(graph_manager, beta, floor(generations), n):
            solved_consistently = True
            # print('success')
        # else:
        #     print('failure')
    lower_bound = n / 2
    upper_bound = n

    n = lower_bound
    solved_consistently = False
    while not solved_consistently and n < upper_bound:
        n *= 1.1
        # print("Solving for n =", round(n), end=' ')
        generations = MAX_EVALUATIONS / n
        if can_solve_10_times(graph_manager, beta, floor(generations), round(n)):
            solved_consistently = True
            # print('success')
        # else:
        #     print('failure')

    return min(upper_bound, round(n))


def optimality_found(graph_manager, beta, generations, population_size, mutation_rate=1, crossover_rate=0.8,
                     iterations=100):
    optimal_count = 0
    for i in range(0, iterations):
        if run_single_experiment(graph_manager, beta, generations, population_size, mutation_rate, crossover_rate):
            optimal_count += 1
    return optimal_count / iterations


def can_solve_10_times(graph_manager, beta, generations, population_size):
    for i in range(0, 10):
        if not run_single_experiment(graph_manager, beta, generations, population_size):
            return False
    return True


def run_single_experiment(graph_manager, beta, generations, population_size, mutation_rate=1, crossover_rate=0.8,
                          verbose=False):
    """Run a single experiment with the provided parameters. Returns whether or not optimality was reached"""
    global evaluations
    evaluations = 0
    global fast_mutation_operator
    global mask

    data = [graph_manager]
    parameter_count = data[0].n

    fast_mutation_operator = FastMutationOperator(parameter_count, beta=beta)
    mask_percent = get_mask_percent(get_relative_neighbours(graph_manager.n, graph_manager.raw_edges))
    mask = get_mask(mask_percent)
    ga = pyeasyga.GeneticAlgorithm(data,
                                   population_size=population_size,
                                   generations=generations,
                                   crossover_probability=crossover_rate,
                                   mutation_probability=mutation_rate,
                                   elitism=True,
                                   maximise_fitness=True)
    ga.fitness_function = fitness
    ga.create_individual = create_individual
    ga.mutate_function = mutate_individual
    ga.run(graph_manager.bkv)

    best_fitness, best_individual = ga.best_individual()

    if verbose:
        print("(", best_fitness, "/", graph_manager.bkv, ")", "eval", evaluations, end=' ')
    return best_fitness == graph_manager.bkv


def crossover(graphs, rates):
    print("Crossover")
    for g in graphs:
        graph_manager = GraphManager("./data/maxcut/" + g, verbose=False)
        results = []
        for crossover_rate in rates:
            population_size = 64
            optimality_percent = optimality_found(graph_manager=graph_manager, beta=1.5,
                                                  generations=MAX_EVALUATIONS // population_size,
                                                  population_size=population_size,
                                                  mutation_rate=1,
                                                  crossover_rate=crossover_rate)
            print(crossover_rate, optimality_percent)
            results.append(optimality_percent)
        print(g, results)
        plt.plot(rates, results)

    plt.xlabel("Crossover rate")
    plt.ylabel("Optimality Percent")
    plt.title("Influence of varying crossover rate on optimality percent for different graphs")
    plt.legend(graphs)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig('white_crossover.png')
    plt.clf()


def mutation(graphs, rates):
    print("Mutation")

    for g in graphs:
        graph_manager = GraphManager("./data/maxcut/" + g, verbose=False)
        results = []
        for mutation_rate in rates:
            population_size = 64
            optimality_percent = optimality_found(graph_manager=graph_manager, beta=1.5,
                                                  generations=MAX_EVALUATIONS // population_size,
                                                  population_size=population_size,
                                                  mutation_rate=mutation_rate,
                                                  crossover_rate=0.8)
            print(mutation_rate, optimality_percent)
            results.append(optimality_percent)
        print(g, results)
        plt.plot(rates, results)

    plt.xlabel("Mutation rate")
    plt.ylabel("Optimality Percent")
    plt.title("Influence of varying mutation rate on optimality percent for different graphs")
    plt.legend(graphs)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig('white_mutation.png')
    plt.clf()


def beta(graphs, betas):
    print("Beta")

    for g in graphs:
        graph_manager = GraphManager("./data/maxcut/" + g, verbose=False)
        results = []
        for beta in betas:
            population_size = 64
            optimality_percent = optimality_found(graph_manager=graph_manager, beta=beta,
                                                  generations=MAX_EVALUATIONS // population_size,
                                                  population_size=population_size,
                                                  mutation_rate=1,
                                                  crossover_rate=0.8)
            print(beta, optimality_percent)
            results.append(optimality_percent)
        print(g, results)
        plt.plot(betas, results)

    plt.xlabel("Beta")
    plt.ylabel("Optimality Percent")
    plt.title("Influence of varying beta on optimality percent for different graphs")
    plt.legend(graphs)
    # plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig('white_beta.png')
    plt.clf()


def population(graphs, population_sizes):
    print("Population")

    for g in graphs:
        graph_manager = GraphManager("./data/maxcut/" + g, verbose=False)
        results = []
        for population_size in population_sizes:
            optimality_percent = optimality_found(graph_manager=graph_manager, beta=1.5,
                                                  generations=MAX_EVALUATIONS // population_size,
                                                  population_size=population_size,
                                                  mutation_rate=1,
                                                  crossover_rate=0.8)
            print(population_size, optimality_percent)
            results.append(optimality_percent)
        print(g, results)
        plt.plot(population_sizes, results)

    plt.xlabel("Population Size")
    plt.ylabel("Optimality Percent")
    plt.title("Influence of varying population size on optimality percent for different graphs")
    plt.legend(graphs)
    # plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig('white_population.png')
    plt.clf()


def main():
    global fast_mutation_operator

    graphs = [
        # "set0a/n0000025i00.txt",
        "set0b/n0000025i00.txt",
        "set0c/n0000025i00.txt",
        "set0d/n0000025i00.txt",
        "set0e/n0000025i00.txt",
    ]

    pop_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    betas = [1.01, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 3, 5]
    mutation_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    crossover_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    population(graphs, pop_sizes)
    beta(graphs, betas)
    mutation(graphs, mutation_rates)
    crossover(graphs, crossover_rates)


if __name__ == "__main__":
    main()
