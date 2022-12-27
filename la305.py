import random
import numpy as np
from time import time
from pyeasyga import pyeasyga

# setup seed data

start_time = time()
array = []
bestIndividuals = []

seed_data = \
 [[0.8, 0.3, 0.3, 0.4 ,0.2, 0.5, 0.8, 0.5, 1.0, 0.3, 0.6, 0.2 ,0.2, 0.9, 0.2, 0.1, 0.7, 0.5,
  0.2, 0.6, 0.7 ,0.8, 0.8 ,1.0, 0.7, 0.2, 0.8 ,0.9, 0.5 ,0.5],
 [0.7, 0.7, 0.1, 0.3, 0.3, 0.9, 0.4, 0.3, 0.6 ,0.5, 0.2, 0.2 ,0.8, 0.9, 0.2, 0.3, 0.4, 0.5,
  0.2, 1.0, 0.2, 0.5, 0.4, 0.6, 0.2, 0.5, 0.3, 0.7, 1.0, 0.5],
 [0.6, 0.4, 0.8, 0.5, 0.7, 0.3, 0.8, 0.5, 0.8, 0.9, 0.7, 1.0, 0.3, 0.4, 0.9, 0.9, 0.3,0.2,
  1.0, 0.7, 0.9, 0.3, 0.4, 0.5, 0.3, 1.0, 0.1, 0.4, 0.6, 0.5],
 [0.8, 0.9, 0.5, 0.2, 0.6, 0.6, 0.4, 0.7, 0.9, 0.2, 0.8, 0.3 ,0.3, 0.6, 0.9, 0.5, 0.6 ,0.4,
  0.5, 0.7, 0.7, 1.0, 0.9, 1.0, 0.8, 0.7, 0.4, 0.6, 0.3, 0.2],
 [0.1, 0.8, 0.2, 0.5, 0.3, 0.9, 0.6, 0.3, 0.3, 0.9, 0.9, 0.7 ,0.7, 0.4, 0.5 ,0.4, 0.4 ,0.6,
  0.5, 0.3, 0.5, 0.9, 0.1, 0.8, 0.9, 0.1, 0.4, 0.6, 0.7, 0.5]]

stask = [["s0", "s1"], ["s0", "s1", "s2"], ["s0", "s1"], ["s0", "s1", "s2"], ["s0", "s1", "s2"], ["s0", "s1"],
         ["s0", "s1", "s2"], ["s0", "s1", "s2"], ["s0", "s1"],
         ["s0", "s1", "s2"], ["s0", "s1", "s2"], ["s0", "s1"], ["s0", "s1", "s2"], ["s0", "s1"], ["s0", "s1"],
         ["s0", "s1"], ["s0", "s1", "s2"], ["s0", "s1", "s2"],
         ["s0", "s1"], ["s0", "s1", "s2"], ["s0", "s1", "s2"], ["s0", "s1"],
         ["s0", "s1", "s2"], ["s0", "s1", "s2"], ["s0", "s1", "s2"], ["s0", "s1", "s2"], ["s0", "s1"], ["s0", "s1"],
         ["s0", "s1", "s2"], ["s0", "s1"]]

semp = [["s0", "s1", "s2", "s3"], ["s0", "s1", "s2", "s3", "s4"], ["s0", "s1", "s2", "s3"],
        ["s0", "s1", "s2", "s3", "s4"], ["s0", "s1", "s2", "s3"],
        ]

effortTask = [
    [4, 6, 20, 7, 8, 19, 5, 10, 10, 7, 12, 12, 14, 7, 11, 4, 4, 11, 8, 21, 25, 20, 5, 12, 6, 10, 12, 17, 12, 4]]
salary = np.array([[8677.79053139821, 8677.79053139821, 8677.79053139821, 8677.79053139821, 8677.79053139821,
                    8677.79053139821, 8677.79053139821, 8677.79053139821, 8677.79053139821, 8677.79053139821,
                    8677.79053139821, 8677.79053139821, 8677.79053139821, 8677.79053139821, 8677.79053139821,
                    8677.79053139821, 8677.79053139821, 8677.79053139821, 8677.79053139821, 8677.79053139821,
                    8677.79053139821, 8677.79053139821, 8677.79053139821, 8677.79053139821, 8677.79053139821,
                    8677.79053139821, 8677.79053139821, 8677.79053139821, 8677.79053139821, 8677.79053139821],
                   [10655.931583737452, 10655.931583737452, 10655.931583737452, 10655.931583737452, 10655.931583737452,
                    10655.931583737452, 10655.931583737452, 10655.931583737452, 10655.931583737452, 10655.931583737452,
                    10655.931583737452, 10655.931583737452, 10655.931583737452, 10655.931583737452, 10655.931583737452,
                    10655.931583737452, 10655.931583737452, 10655.931583737452, 10655.931583737452, 10655.931583737452,
                    10655.931583737452, 10655.931583737452, 10655.931583737452, 10655.931583737452, 10655.931583737452,
                    10655.931583737452, 10655.931583737452, 10655.931583737452, 10655.931583737452, 10655.931583737452],
                   [10473.793087484073, 10473.793087484073, 10473.793087484073, 10473.793087484073, 10473.793087484073,
                    10473.793087484073, 10473.793087484073, 10473.793087484073, 10473.793087484073, 10473.793087484073,
                    10473.793087484073, 10473.793087484073, 10473.793087484073, 10473.793087484073, 10473.793087484073,
                    10473.793087484073, 10473.793087484073, 10473.793087484073, 10473.793087484073, 10473.793087484073,
                    10473.793087484073, 10473.793087484073, 10473.793087484073, 10473.793087484073, 10473.793087484073,
                    10473.793087484073, 10473.793087484073, 10473.793087484073, 10473.793087484073, 10473.793087484073],
                   [9088.30981299338, 9088.30981299338, 9088.30981299338, 9088.30981299338, 9088.30981299338,
                    9088.30981299338, 9088.30981299338, 9088.30981299338, 9088.30981299338, 9088.30981299338,
                    9088.30981299338, 9088.30981299338, 9088.30981299338, 9088.30981299338, 9088.30981299338,
                    9088.30981299338, 9088.30981299338, 9088.30981299338, 9088.30981299338, 9088.30981299338,
                    9088.30981299338, 9088.30981299338, 9088.30981299338, 9088.30981299338, 9088.30981299338,
                    9088.30981299338, 9088.30981299338, 9088.30981299338, 9088.30981299338, 9088.30981299338],
                   [9333.91392104519, 9333.91392104519, 9333.91392104519, 9333.91392104519, 9333.91392104519,
                    9333.91392104519, 9333.91392104519, 9333.91392104519, 9333.91392104519, 9333.91392104519,
                    9333.91392104519, 9333.91392104519, 9333.91392104519, 9333.91392104519, 9333.91392104519,
                    9333.91392104519, 9333.91392104519, 9333.91392104519, 9333.91392104519, 9333.91392104519,
                    9333.91392104519, 9333.91392104519, 9333.91392104519, 9333.91392104519, 9333.91392104519,
                    9333.91392104519, 9333.91392104519, 9333.91392104519, 9333.91392104519, 9333.91392104519]])


allocation = []



# initialise the GA
ga = pyeasyga.GeneticAlgorithm(seed_data,
                               population_size=64,
                               generations=5064,
                               crossover_probability=0.8,
                               mutation_probability=0.2,
                               elitism=True,
                               maximise_fitness=False)


# define and set function to create a candidate solution representation
def create_individual(data):
    allocation = np.zeros(shape=(len(semp), len(stask)))
    individual = []
    for row, employee in enumerate(semp):
        for column, task in enumerate(stask):
            score = 0
            employee_skills = set(employee)
            task_skills = set(task)
            result = employee_skills.intersection(task_skills)
            if len(result) == len(task_skills):
                score = random.uniform(0.1, 1)
            allocation[row, column] = score
            individual = allocation
    return individual


np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

firstIndividual = create_individual([])
secondIndividualForCrossOver = create_individual([])

ga.create_individual = create_individual


# define and set the GA's crossover operation
def crossover(parent_1, parent_2):
    # Logic for CrossOver
    mutate_index1 = parent_1[:, 0:15]
    mutate_index2 = parent_1[:, 15:30]
    mutate_index_x1 = parent_2[:, 0:15]
    mutate_index_x2 = parent_2[:, 15:30]
    crossOverResult1 = np.concatenate((mutate_index_x2, mutate_index1), axis=1)
    crossOverResult2 = np.concatenate((mutate_index2, mutate_index_x1), axis=1)
    return crossOverResult1, crossOverResult2


ga.crossover_function = crossover


# define and set the GA's mutation operation
def mutate(individual):
    mutate_index1 = individual[:, 0:15]
    # Logic for Mutation
    mutate_index2 = individual[:, 15:30]
    mutationResult = np.concatenate((mutate_index2, mutate_index1), axis=1)
    return mutationResult


mutationResult = mutate(firstIndividual)

ga.mutate_function = mutate


# define and set the GA's selection operation
def selection(population):
    res = random.choice(population)
    new_fit = []
    best_ind = []
    for i in range(ga.population_size):
        new_fit.append(res.fitness)
        best_ind.append(res)
        new_fit.sort(reverse=True)
    array.append(new_fit[0:10][0])  # append fitness for each generation into array
    bestIndividuals.append(best_ind[0:10])  # append fitness for each generation into array

    return random.choice(population)


ga.selection_function = selection


# CHECKING IF NEW MATRIX DOES NOT HAVE ANY COLUMN WITH ALL ZERO VALUES
def doesMatrixPassConstraintOne(matrix):
    constrainArr = np.array([0, 0, 0, 0, 0]).reshape(5, 1)
    output = matrix == constrainArr
    if output[:, 0:1].all():
        return False
    return True


# CHECKING IF NEW MATRIX MAINTAINS OLD MATRIX STATE I.E O REMAINS O AND VICE VERSA
def doesMatrixPassConstraintTwo(resultMatrix, oldMatrix):
    for row, individual in enumerate(resultMatrix):
        for column, col in enumerate(oldMatrix):
            if resultMatrix[row][column] > 0 and oldMatrix[row][column] > 0 or resultMatrix[row][column] == 0 and \
                    oldMatrix[row][column] == 0:
                continue
            else:
                return False
    return True


def project_duration(allocation):
    add_column = np.sum(allocation, axis=0)
    add_column = np.array([add_column])
    division_result = np.divide(effortTask, add_column)
    final_duration = np.sum(division_result)

    return final_duration, division_result


def project_cost(allocation , division_result):
    cost = np.multiply(allocation, salary)
    duration_matrix = np.repeat(division_result, repeats=5, axis=0)
    pre_final_cost = np.multiply(cost, duration_matrix)
    final_cost = (np.sum(pre_final_cost))

    return final_cost


def fitness(allocation, data):


    passedFirstValidation = doesMatrixPassConstraintOne(mutationResult)
    passedSecondValidation = doesMatrixPassConstraintTwo(mutationResult, firstIndividual)

    if not (passedFirstValidation and passedSecondValidation):
        allocation = create_individual([])

    result = project_duration(allocation)
    final_cost = project_cost(allocation, result[1])

    total_fitness = 1 / ((0.00001 * final_cost) + (0.1 * result[0]))
    return total_fitness


ga.fitness_function = fitness  # set the GA's fitness function
ga.run()  # run the GA

# end_time = time()

# print("RESULTS FROM GENETIC ALGORITHM .....")
# # print("TIME TAKEN ", "{:.2f}".format(end_time - start_time) + " seconds")
# print("BEST INDIVIDUAL FITNESS FROM GENETIC ALGORITHM IS: ", "{:.4f}".format(ga.best_individual()[0]))
# print("BEST INDIVIDUAL ARRAY: ")
# print(ga.best_individual()[1])

print("RESULTS FROM LATE ACCEPTANCE .....")
start_index = 0
size = ga.population_size
# SECTION ARRAY WITH ALL POSSIBLE FITNESS USING POULATION_SIZE, SELECT BEST SET FITNESS COMPARING CURRENT BEST SET TO THE NEXT SET FROM NEXT GENERATION
for i in range(int(len(array) / size)):
    start_index = start_index + size
    end_index = start_index + size
    if i == 0:
        start_index = 0
        end_index = size
        best_fitness = array[start_index:end_index]
    next_set = array[end_index:end_index + size]
    best_fitness.sort(reverse=True)
    next_set.sort(reverse=True)
    if end_index < len(array):
        for index, value in enumerate(best_fitness):
            if next_set[index] > best_fitness[index]:
                best_fitness[index] = next_set[index]


# GET BEST INDIVIDUAL FROM LATE ACCEPTANCE
def get_best_individual(results):
    for index, value in enumerate(results):
        if value[0].fitness == np.amax(array):
            return value[0].genes


# REMOVE FITNESS APPEARING MORE THAN ONCE
def get_unique_fitness(numbers):
    list_of_unique_fitness = []

    unique_numbers = set(numbers)

    for number in unique_numbers:
        list_of_unique_fitness.append(number)
    return list_of_unique_fitness


unique_best_fitness = get_unique_fitness(best_fitness)
unique_best_fitness.sort(reverse=True)
best_20_fitness = unique_best_fitness[0:20]
end_time = time()

# print("TOP 10 BEST FITNESS FROM LATE ACCEPTANCE IS: ", unique_best_fitness[0:20])
print("BEST INDIVIDUAL IS FROM LATE ACCEPTANCE ", get_best_individual(bestIndividuals))
# print("BEST OVERALL FITNESS FROM LATE ACCEPTANCE IS: ", np.amax(unique_best_fitness))
# print("BEST OVERALL FITNESS FROM LATE ACCEPTANCE USING THE INITIAL ARRAY WITH ALL FITNESS FROM ALL GENERATIONS IS: ",
#       np.amax(array))
# print("TWO VALUES ARE EQUAL ",
#       np.amax(unique_best_fitness) == np.amax(array))  # THIS IS TO VET THE ACCURACY OF OUR ANSWER
print("THE BEST FITNESS IS ", best_20_fitness[0])
print("TIME TAKEN ", "{:.2f}".format(end_time - start_time) + " seconds")


result = project_duration(get_best_individual(bestIndividuals))
print("PROJECT DURATION ", result[0])
proj_cost = project_cost(get_best_individual(bestIndividuals), result[1])
print("PROJECT COST ", proj_cost)