import array
import random
import sys
import multiprocessing

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

################################# PARAMETERS #################################

NUM_MOVES = 200
NUM_FORAGING = 100
NUM_ENVIRONMENT_ROWS = NUM_ENVIRONMENT_COLUMNS = NUM_ENVIRONMENT_SLICES = 10

DOWN_BASE = 729  # 3^5
UP_BASE = 243  # 3^5
NORTH_BASE = 81  # 3^4
SOUTH_BASE = 27  # 3^3
EAST_BASE = 9  # 3^2
WEST_BASE = 3  # 3^1
CENTER_BASE = 1  # 3^0

GREEN = 1  # analogously to CAN cell
RED = 0  # analogously to EMPTY cell
WALL = 2

WALL_PENALTY = -5

MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_EAST = 2
MOVE_WEST = 3
MOVE_UP = 4
MOVE_DOWN = 5
STAY_PUT = 6
PICK_UP = 7
RANDOM_MOVE = 8

WATER_COLOUR_BASE = 2187
STARTING_COORDINATES = 5

FITNESS_POINTS_PER_QUANTITY = {
    0: 0,
    1: 1,
    2: 3,
    3: 6,
    4: 9,
    5: 10,
    6: 9,
    7: 6,
    8: 3,
    9: 1,
    10: 0
}

environment = 0

##############################################################################

robby_row = robby_column = robby_cubeslice = STARTING_COORDINATES


# Returns random integers from lower_bound (inclusive) to upper_bound (inclusive).
# A helper function.
def randnr(lower_bound, upper_bound):
    upper_bound += 1
    return np.random.randint(low=lower_bound, high=upper_bound)


# Returns an environment value:
#     water quantities: 0-10
def environment_values():
    return randnr(0, 10)


# Environment class, basically a matrix.
class Environment:
    def __init__(self, init=True):
        self.m = NUM_ENVIRONMENT_ROWS + 2
        self.n = NUM_ENVIRONMENT_COLUMNS + 2
        self.s = NUM_ENVIRONMENT_SLICES + 2
        self.cube = np.zeros((self.s, self.n, self.m), dtype=int)

        for slices in range(0, self.s):
            if slices == 0 or slices == self.s - 1:
                self.cube[slices] = np.full((self.n, self.m), -WALL, dtype=int)
                continue
            for i in range(0, self.n):
                for j in range(0, self.m):
                    if i == 0 or i == self.n - 1 or j == 0 or j == self.m - 1:
                        self.cube[slices][i][j] = -WALL
                    else:
                        self.cube[slices][i][j] = environment_values()

    def __setValue__(self, idx_slice, idx_row, idx_col, value):
        self.cube[idx_slice][idx_row][idx_col] = value

    def __getValue__(self, idx_slice, idx_row, idx_col):
        return self.cube[idx_slice][idx_row][idx_col]

    def __setitem__(self, idx_slice, value):
        self.cube[idx_slice] = value

    def __getitem__(self, idx_slice):
        return self.cube[idx_slice]

    def __printEnvironment__(self):
        for slices in range(0, self.s):
            print(np.matrix(self.cube[slices]))


# Calculates the state of a cube cell.
def calculate_state_one_cell(cubeslice, row, column, base, individual):
    global environment

    if ((row <= 0) or (row >= (NUM_ENVIRONMENT_ROWS + 1))) or (
            (column <= 0) or (column >= (NUM_ENVIRONMENT_COLUMNS + 1))) or (
            (cubeslice <= 0) or (cubeslice >= NUM_ENVIRONMENT_SLICES)):
        return base * WALL
    else:
        water_quantity = environment[cubeslice][row][column]
        # if the gene associated with the amount of water is even then Robby sees "red" otherwise it sees it as a "green" zone
        if individual[WATER_COLOUR_BASE + water_quantity] % 2 == 1:
            return base * GREEN
        return base * RED


# Performs an action and returns the reward for it.
def perform_action(action, individual):
    global robby_cubeslice, robby_row, robby_column, FITNESS_POINTS_PER_QUANTITY, environment
    reward = 0

    # NORTH
    n_row = robby_row - 1
    n_column = robby_column

    if action == MOVE_NORTH:
        if ((n_row <= 0) or (n_row >= (NUM_ENVIRONMENT_ROWS + 1))) or (
                (n_column <= 0) or (n_column >= (NUM_ENVIRONMENT_COLUMNS + 1))) or (
                (robby_cubeslice <= 0) or (robby_cubeslice >= NUM_ENVIRONMENT_SLICES)):
            reward = WALL_PENALTY
        else:
            robby_row = robby_row - 1

    # SOUTH
    n_row = robby_row + 1
    n_column = robby_column

    if action == MOVE_SOUTH:
        if ((n_row <= 0) or (n_row >= (NUM_ENVIRONMENT_ROWS + 1))) \
                or ((n_column <= 0) or (n_column >= (NUM_ENVIRONMENT_COLUMNS + 1))) \
                or ((robby_cubeslice <= 0) or (robby_cubeslice >= NUM_ENVIRONMENT_SLICES)):
            reward = WALL_PENALTY
        else:
            robby_row = robby_row + 1

    # EAST
    n_row = robby_row
    n_column = robby_column + 1

    if action == MOVE_EAST:
        if ((n_row <= 0) or (n_row >= (NUM_ENVIRONMENT_ROWS + 1))) \
                or ((n_column <= 0) or (n_column >= (NUM_ENVIRONMENT_COLUMNS + 1))) \
                or ((robby_cubeslice <= 0) or (robby_cubeslice >= NUM_ENVIRONMENT_SLICES)):
            reward = WALL_PENALTY
        else:
            robby_column = robby_column + 1

    # WEST
    n_row = robby_row
    n_column = robby_column - 1

    if action == MOVE_WEST:
        if ((n_row <= 0) or (n_row >= (NUM_ENVIRONMENT_ROWS + 1))) \
                or ((n_column <= 0) or (n_column >= (NUM_ENVIRONMENT_COLUMNS + 1))) \
                or ((robby_cubeslice <= 0) or (robby_cubeslice >= NUM_ENVIRONMENT_SLICES)):
            reward = WALL_PENALTY
        else:
            robby_column = robby_column - 1

    if action == STAY_PUT:
        None  # do nothing

    if action == PICK_UP:
        water_quantity = environment[robby_cubeslice][robby_row][robby_column]
        reward = FITNESS_POINTS_PER_QUANTITY[water_quantity]
        environment[robby_cubeslice][robby_row][robby_column] = 0

    if action == RANDOM_MOVE:
        random_step = np.random.randint(low=0, high=6)
        # steps randomly --> 0,1,2,3  # 4 up, 5 down
        reward = perform_action(random_step, individual)

    ### 3D moveset

    # UP
    if action == MOVE_UP:
        n_cubeslice = robby_cubeslice + 1
        if ((robby_row <= 0) or (robby_row >= (NUM_ENVIRONMENT_ROWS + 1))) \
                or ((robby_column <= 0) or (robby_column >= (NUM_ENVIRONMENT_COLUMNS + 1))) \
                or ((n_cubeslice <= 0) or (n_cubeslice >= NUM_ENVIRONMENT_SLICES)):
            reward = WALL_PENALTY
        else:
            robby_cubeslice += 1
    # DOWN
    if action == MOVE_DOWN:
        n_cubeslice = robby_cubeslice - 1
        if ((robby_row <= 0) or (robby_row >= (NUM_ENVIRONMENT_ROWS + 1))) \
                or ((robby_column <= 0) or (robby_column >= (NUM_ENVIRONMENT_COLUMNS + 1))) \
                or ((n_cubeslice <= 0) or (n_cubeslice >= NUM_ENVIRONMENT_SLICES)):
            reward = WALL_PENALTY
        else:
            robby_cubeslice -= 1
    return reward


# Calculates the overall fitness of an individual.
# It forages NUM_FORAGING times and in each session
# it can perform an action NUM_MOVES times.
def calculate_individual_fitness(individual):
    global robby_cubeslice, robby_row, robby_column, environment
    fitness_points = 0

    for foraging_number in range(1, NUM_FORAGING + 1):
        robby_row = 5
        robby_column = 5
        robby_cubeslice = 5
        environment = Environment()
        for move_number in range(1, NUM_MOVES + 1):
            state = 0

            state += calculate_state_one_cell(cubeslice=robby_cubeslice, row=robby_row, column=robby_column,
                                              base=CENTER_BASE, individual=individual)
            state += calculate_state_one_cell(cubeslice=robby_cubeslice, row=robby_row - 1, column=robby_column,
                                              base=NORTH_BASE, individual=individual)
            state += calculate_state_one_cell(cubeslice=robby_cubeslice, row=robby_row, column=robby_column + 1,
                                              base=EAST_BASE, individual=individual)
            state += calculate_state_one_cell(cubeslice=robby_cubeslice, row=robby_row + 1, column=robby_column,
                                              base=SOUTH_BASE, individual=individual)
            state += calculate_state_one_cell(cubeslice=robby_cubeslice, row=robby_row, column=robby_column - 1,
                                              base=WEST_BASE, individual=individual)
            state += calculate_state_one_cell(cubeslice=robby_cubeslice + 1, row=robby_row, column=robby_column,
                                              base=UP_BASE, individual=individual)
            state += calculate_state_one_cell(cubeslice=robby_cubeslice - 1, row=robby_row, column=robby_column,
                                              base=DOWN_BASE, individual=individual)

            action = individual[state - 1]
            fitness_points = fitness_points + perform_action(action, individual)

    return fitness_points / NUM_FORAGING,


toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMax)
toolbox.register("attributes", random.randint, 0, 8)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attributes, 2187 + 11)


# main function
def main():
    global toolbox

    pool = multiprocessing.Pool(processes=32)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=8, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", calculate_individual_fitness)
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    random.seed(170)
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 10, stats=stats, halloffame=hof, verbose=True)

    for h in hof:
        print(h)


if __name__ == '__main__':
    main()
