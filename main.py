import array
import random
import sys

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

################################# PARAMETERS #################################

NUM_MOVES = 20
NUM_FORAGING = 100
NUM_ENVIRONMENT_ROWS = NUM_ENVIRONMENT_COLUMNS = 10
CAN_DENSITY = 0.5

NORTH_BASE = 81  # 3^4
SOUTH_BASE = 27  # 3^3
EAST_BASE = 9  # 3^2
WEST_BASE = 3  # 3^1
CENTER_BASE = 1  # 3^0

EMPTY = 0
CAN = 1
WALL = 2

WALL_PENALTY = -5
CAN_REWARD = 10
CAN_PENALTY = -1

MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_EAST = 2
MOVE_WEST = 3
STAY_PUT = 4
PICK_UP = 5
RANDOM_MOVE = 6

STARTING_COORDINATES = 1

##############################################################################

robby_row = robby_column = STARTING_COORDINATES


# Returns random integers from lower_bound (inclusive) to upper_bound (inclusive).
# A helper function.
def randnr(lower_bound, upper_bound):
    upper_bound += 1
    return numpy.random.randint(low=lower_bound, high=upper_bound)


# Returns an environment value:
#     CAN    with a probability of CAN_DENSITY
#     EMPTY  with a probability of 1 - CAN_DENSITY
def environment_values():
    if randnr(1, 10) < (CAN_DENSITY * 10):
        return CAN
    return EMPTY


# Environment class, basically a matrix.
class Environment:
    def __init__(self, init=True):
        self.m = NUM_ENVIRONMENT_ROWS + 1
        self.n = NUM_ENVIRONMENT_COLUMNS + 1
        self.rows = [[0] * self.n for x in range(self.m)]
        if init:
            for i in range(0, 10):
                for j in range(0, 10):
                    if i == 0 or i == self.n or j == 0 or j == self.m:
                        self.rows[i][j] = 2
                    else:
                        self.rows[i][j] = environment_values()

    def __setValue__(self, idx_row, idx_col, value):
        self.rows[idx_row][idx_col] = value

    def __getValue__(self, idx_row, idx_col):
        return self.rows[idx_row][idx_col]

    def __setitem__(self, idx_row, value):
        self.rows[idx_row] = value

    def __getitem__(self, idx_row):
        return self.rows[idx_row]

    def __printEnvironment__(self):
        print(numpy.matrix(self.rows))

    def __hasCan__(self, idx_row, idx_col):
        return True if self.__getValue__(idx_row, idx_col) == 1 else False


# Calculates state of a quarter.
def calc_state_one_quarter(environment, row, column, base):
    if ((row <= 0) or (row >= NUM_ENVIRONMENT_ROWS)) or (
            (column <= 0) or (column >= NUM_ENVIRONMENT_COLUMNS)):
        return base * WALL
    else:
        if environment.__hasCan__(column, row):
            return base * CAN
        return base * EMPTY


# Calculates the overall fitness of an individual.
# It forages NUM_FORAGING times and in each session
# it can perform an action NUM_MOVES times.
def calc_fitness_one_environment(individual):
    global robby_row, robby_column
    fitness_points = 0

    for foraging_number in range(1, NUM_FORAGING + 1):
        robby_row = 1
        robby_column = 1
        environment = Environment()
        for move_number in range(1, NUM_MOVES + 1):
            state = 0

            state += calc_state_one_quarter(environment=environment, row=robby_row - 1, column=robby_column,
                                            base=NORTH_BASE)
            state += calc_state_one_quarter(environment=environment, row=robby_row, column=robby_column + 1,
                                            base=EAST_BASE)
            state += calc_state_one_quarter(environment=environment, row=robby_row + 1, column=robby_column,
                                            base=SOUTH_BASE)
            state += calc_state_one_quarter(environment=environment, row=robby_row, column=robby_column - 1,
                                            base=WEST_BASE)

            action = individual[state]
            fitness_points = fitness_points + perform_action(action, environment)

    return fitness_points / NUM_FORAGING,


# Performs an action and returns the reward for it.
def perform_action(action, environment):
    global robby_row, robby_column
    reward = 0

    # NORTH
    n_row = robby_row - 1
    n_column = robby_column

    if action == MOVE_NORTH:
        if (((n_row <= 0) or (n_row >= NUM_ENVIRONMENT_ROWS))
                or ((n_column <= 0) or (n_column >= NUM_ENVIRONMENT_COLUMNS))):
            reward = WALL_PENALTY
        else:
            robby_row = robby_row - 1

    # SOUTH
    n_row = robby_row + 1
    n_column = robby_column

    if action == MOVE_SOUTH:
        if (((n_row <= 0) or (n_row >= NUM_ENVIRONMENT_ROWS))
                or ((n_column <= 0) or (n_column >= NUM_ENVIRONMENT_COLUMNS))):
            reward = WALL_PENALTY
        else:
            robby_row = robby_row + 1

    # EAST
    n_row = robby_row
    n_column = robby_column + 1

    if action == MOVE_EAST:
        if (((n_row <= 0) or (n_row >= NUM_ENVIRONMENT_ROWS))
                or ((n_column <= 0) or (n_column >= NUM_ENVIRONMENT_COLUMNS))):
            reward = WALL_PENALTY
        else:
            robby_column = robby_column + 1

    # WEST
    n_row = robby_row
    n_column = robby_column - 1

    if action == MOVE_WEST:
        if (((n_row <= 0) or (n_row >= NUM_ENVIRONMENT_ROWS))
                or ((n_column <= 0) or (n_column >= NUM_ENVIRONMENT_COLUMNS))):
            reward = WALL_PENALTY
        else:
            robby_column = robby_column - 1

    if action == STAY_PUT:
        None  # do nothing

    if action == PICK_UP:
        if environment.__hasCan__(robby_row, robby_column):
            reward = CAN_REWARD
            environment.__setValue__(robby_row, robby_column, 0)
        else:
            reward = CAN_PENALTY

    if action == RANDOM_MOVE:
        random_step = numpy.random.randint(low=0, high=4)
        # steps randomly --> 0,1,2,3
        reward = perform_action(random_step, environment)

    return reward


# Main function.
def main():
    random.seed(169)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attributes", random.randint, 0, 6)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attributes, 243)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=6, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", calc_fitness_one_environment)

    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 1000, stats=stats, halloffame=hof, verbose=True)

    return pop, stats, hof


if __name__ == "__main__":
    # sys.stdout = open('resultsFile.txt', mode='w', buffering=1)
    main()
