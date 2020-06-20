import random

from pyeasyga import pyeasyga

from utils.graph_manager import GraphManager
from fastga import FastMutationOperator

evaluations = 0


def fitness(individual, data):
    global evaluations
    evaluations += 1

    fitness_score = data[0].evaluate_individual(individual)
    return fitness_score


def create_individual(data):
    parameter_count = data[0].n
    return [random.randint(0, 1) for _ in range(parameter_count)]


def mutate_individual(individual):
    fast_mutation_operator.mutate(individual, inplace=True)


def main():
    global fast_mutation_operator

    data = [GraphManager('../data/maxcut/set0a/n0000100i99.txt', verbose=True)]
    parameter_count = data[0].n

    fast_mutation_operator = FastMutationOperator(parameter_count, beta=1.01)
    ga = pyeasyga.GeneticAlgorithm(data,
                                   population_size=10,
                                   generations=100,
                                   crossover_probability=0.8,
                                   mutation_probability=0.5,
                                   elitism=False,
                                   maximise_fitness=True)
    ga.fitness_function = fitness
    ga.create_individual = create_individual
    ga.mutate_function = mutate_individual
    ga.run()
    print(ga.best_individual())
    print("Evaluations", evaluations)


if __name__ == "__main__":
    main()
