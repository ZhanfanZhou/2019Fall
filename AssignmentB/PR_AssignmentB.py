import copy
import numpy as np
import random

years_costs = [[0.5, 1.0, 1.5, 0.1], [0.3, 0.8, 1.5, 0.4], [0.2, 0.2, 0.3, 0.1]]
rewards = [0.2, 0.3, 0.5, 0.1]
limitation = [3.1, 2.5, 0.4]
MUTATION_RATE = 0.03
STOP_THRESHOLD = 66
ENCODE_LENGTH = 4
CROSSOVER_POINT = 2  # index where switch gene


def check_constrain(individual):  # check a gene whether meets the constrains
    for year in range(len(years_costs)):
        if sum(map(lambda (w, x): w*x, zip(individual, years_costs[year]))) > limitation[year]:
            print individual, "year %s fail" % str(year+1)  # year starts from 1
            return False
    # print "total reward: ", get_fitness(individual)
    return True


def get_fitness(individual):
    if individual == [1, 0, 0, 1]:  # REALLY DON'T KNOW WHY it returns 0.3000000004 instead of 0.3 in this situation
        return 0.3
    return sum(map(lambda (w, x): w * x, zip(individual, rewards)))


def rank_fitness(population_alive):  # rank the fitness and return the highest value
    ranked_reward = []
    for individual in population_alive:
        ranked_reward.append((individual, get_fitness(individual)))
    ranked_reward.sort(key=lambda x: x[1], reverse=True)
    return ranked_reward[0]


def create_first_generation():  # random choice first generation
    return [np.random.randint(0, 2, ENCODE_LENGTH).tolist(), np.random.randint(0, 2, ENCODE_LENGTH).tolist()]


def crossover(population_alive):  # reproduce offspring, return all the new offspring
    if len(population_alive) == 0:  # if the alive is not enough to reproduce next generation, randomly add one or two
        print "0 gene left! auto add two"
        return create_first_generation()
    else:
        while len(population_alive) == 1:
            print "only 1 gene left! auto add one"
            population_alive.append(np.random.randint(0, 2, ENCODE_LENGTH).tolist())
            population_alive = clean_population(population_alive)
        offsprings = []
        for individual_id in range(len(population_alive)-1):  # Reproduction
            for not_cross in range(individual_id+1, len(population_alive)):
                offspring_1 = population_alive[individual_id][:CROSSOVER_POINT]+population_alive[not_cross][CROSSOVER_POINT:]
                offspring_2 = population_alive[not_cross][:CROSSOVER_POINT] + population_alive[individual_id][CROSSOVER_POINT:]
                offsprings.append(mutation(offspring_1))
                offsprings.append(mutation(offspring_2))
        print "new born: ", offsprings
        return offsprings


def fitness_selection(population_alive):  # selection process
    new_population = []
    for individual in population_alive:
        if check_constrain(individual):  # check it whether meets the constrains
            new_population.append(individual)
    while not new_population:  # if all genes are removed
        print "no population left, auto add two:"
        for new_born in create_first_generation():
            print "new_born: ", new_born
            if check_constrain(new_born):
                new_population.append(new_born)
    return new_population


def mutation(offspring):  # to decide whether the offspring is to mutate
    copy_ = copy.copy(offspring)
    for i in range(0, ENCODE_LENGTH):
        if random.uniform(0, 10) < 10*MUTATION_RATE:
            offspring[i] = offspring[i] ^ 1
    if copy_ != offspring:
        print copy_, " --MUTATION!--> ", offspring
    return offspring


def clean_population(population_alive):  # delete the same gene in the population
    cleaned = []
    for individual in population_alive:
        if individual not in cleaned:
            cleaned.append(individual)
    return cleaned


repeat = 0  # record how many time the max value occurs
max_val = ([], -1)  # data structure of optimized solution, the initial maximum value is -1
population = create_first_generation()  # random choice first generation
while repeat < STOP_THRESHOLD:  # stop criteria
    print "population: ", population
    population = population+crossover(population)
    print "new population: ", population
    population = clean_population(population)  # delete the same DNA in population
    population = fitness_selection(population)  # select the generation
    print "selected: ", [(individual, get_fitness(individual)) for individual in population]
    print "population size: ", len(population)
    new_max = rank_fitness(population)  # rank the selected population
    if new_max[1] > max_val[1]:  # whether renew the optimized value or mot
        repeat = 1  # we have a new highest value
        max_val = new_max
    else:
        repeat += 1
    print "current max: ", max_val
    print "time: ", repeat
