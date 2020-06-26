from itertools import product
import matplotlib.pyplot as plt

from utils.graph_manager import GraphManager


def main():
    graph_manager = GraphManager('data/maxcut/set0a/n0000012i00.txt', verbose=True)

    strings = list(product('01', repeat=graph_manager.n))

    fitness = []
    x = []

    for string in strings:
        solution = graph_manager.evaluate_individual(string)
        fitness.append(solution)
        x.append(int(''.join(string), 2))

    print("Solution size", len(fitness))
    plot(x, fitness)


def plot(x, fitness):
    plt.plot(x, fitness)
    plt.xlabel("Individual bitstring represented as integer")
    plt.ylabel("Fitness")
    plt.title("Fitness per individual for set0a/n0000012i00")
    plt.show()


if __name__ == "__main__":
    main()
