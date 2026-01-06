# ============================================================
# PHASE 6: GENETIC ALGORITHM
# EIGHT QUEENS PROBLEM
# ============================================================

import numpy as np
import random
import matplotlib.pyplot as plt

N = 8
POPULATION_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.1

# Fitness function
def fitness(board):
    conflicts = 0
    for i in range(N):
        for j in range(i + 1, N):
            if board[i] == board[j] or abs(board[i] - board[j]) == j - i:
                conflicts += 1
    return 28 - conflicts  # Max fitness = 28

# Create initial population
def initialize_population():
    return [np.random.permutation(N) for _ in range(POPULATION_SIZE)]

# Selection
def selection(population):
    weights = [fitness(ind) for ind in population]
    return random.choices(population, weights=weights, k=2)

# Crossover
def crossover(parent1, parent2):
    cut = random.randint(0, N - 1)
    child = list(parent1[:cut])
    for gene in parent2:
        if gene not in child:
            child.append(gene)
    return np.array(child)

# Mutation
def mutate(individual):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(N), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

# Genetic Algorithm
population = initialize_population()
best_fitness = []
best_solution = None

for generation in range(GENERATIONS):
    new_population = []
    for _ in range(POPULATION_SIZE):
        p1, p2 = selection(population)
        child = crossover(p1, p2)
        child = mutate(child)
        new_population.append(child)

    population = new_population
    current_best = max(population, key=fitness)
    current_fitness = fitness(current_best)
    best_fitness.append(current_fitness)

    if current_fitness == 28:
        best_solution = current_best
        break

print("Best Solution (Queen positions by row):", best_solution)
print("Best Fitness:", max(best_fitness))

# Convergence Plot
plt.plot(best_fitness)
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("Phase 6: GA Convergence Plot")
plt.show()
