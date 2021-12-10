import array
import random
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns


import numpy as np
from matplotlib.colors import ListedColormap
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

################################# PARAMETERS #################################

MAP_TYPE = 0 # a colour of collected water squares

NUM_MOVES = 5
NUM_FORAGING = 1
NUM_ENVIRONMENT_ROWS = NUM_ENVIRONMENT_COLUMNS = 10
CAN_DENSITY = 0.5

NORTH_BASE = 81  # 3^4
SOUTH_BASE = 27  # 3^3
EAST_BASE = 9  # 3^2
WEST_BASE = 3  # 3^1
CENTER_BASE = 1  # 3^0

GREEN = 1 # analogously to CAN cell
RED =   0 # analogously to EMPTY cell
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

TOTAL_MOVES = 6
POSSIBLE_MOVE_VARIATIONS = 243
WATER_COLOUR_BASE = 243
STARTING_COORDINATES = 4
ROBBY_POSITION_NUMBER = 12
WATER_QUANTITIES = 11

fitness_points_per_quantity = {
    0: 0,
    1: 2,
    2: 3,
    3: 7,
    4: 9,
    5: 10,
    6: 8,
    7: 6,
    8: 5,
    9: 4,
    10: 1
}

##############################################################################

robby_row = robby_column = STARTING_COORDINATES

# Returns random integers from lower_bound (inclusive) to upper_bound (inclusive).
# A helper function.
def randnr(lower_bound, upper_bound):
    upper_bound += 1
    return np.random.randint(low=lower_bound, high=upper_bound)

# Returns an environment value:
#     water quantities: 0-10
def environment_values():
    return randnr(0,10)

# Environment class, basically a matrix.
class Environment:
    def __init__(self, init=True):
        self.m = NUM_ENVIRONMENT_ROWS + 2
        self.n = NUM_ENVIRONMENT_COLUMNS + 2
        self.rows = [[0] * self.n for x in range(self.m)]
        if init:
            for i in range(0, self.n):
                for j in range(0, self.m):
                    if i == 0 or i == self.n-1 or j == 0 or j == self.m-1:
                        self.rows[i][j] = -WALL
                    else:
                        self.rows[i][j] = environment_values()

    def __setValue__(self, idx_row, idx_col, value):
        self.rows[idx_row][idx_col] = value

    def __setMatrix__(self, matrix):
        self.rows = matrix

    def __getValue__(self, idx_row, idx_col):
        return self.rows[idx_row][idx_col]

    def __setitem__(self, idx_row, value):
        self.rows[idx_row] = value

    def __getitem__(self, idx_row):
        return self.rows[idx_row]

    def __printEnvironment__(self):
        print(np.matrix(self.rows))

    def __getEnvironmentMatrix__(self):
        return np.matrix(self.rows)    

    # just printing out plain map
    def __drawScene__(self):
        matrix = np.squeeze(np.asarray(np.matrix(self.rows)))
        sns.heatmap(matrix, annot=True, square=True, cbar=False, vmin=-2, vmax=10, center= 5, cmap= 'Blues', xticklabels="", yticklabels="")

    # printing out map + robby on it
    def __drawFrame__(self, idx_row, idx_col):
        value = self.rows[idx_row][idx_col]
        self.rows[idx_row][idx_col] = ROBBY_POSITION_NUMBER
        matrix = np.squeeze(np.asarray(np.matrix(self.rows)))
        self.rows[idx_row][idx_col] = value
        
        return sns.heatmap(matrix, annot=True, square=True, cbar=False, vmin=-2, vmax=12, center= 5, cmap= 'Blues', xticklabels="", yticklabels="")

    # printing map percepted by robby + robby
    def __drawPerceptedFrame__(self, idx_row, idx_col, water_configuration):
        value = self.rows[idx_row][idx_col]
        self.rows[idx_row][idx_col] = ROBBY_POSITION_NUMBER
        matrix = np.squeeze(np.asarray(np.matrix(self.rows)))
        self.rows[idx_row][idx_col] = value

        colour_list = ["lightyellow"] # for walls
        colour_list.append("lightgrey") # for collected water cells
        for water_colour in range(len(water_configuration)):
            # even --> red 
            # odd  --> green
            if water_configuration[water_colour] % 2 == 1:
                colour_list.append("red")
            else:
                colour_list.append("lightgreen")
        
        colour_list.append("darkblue") # for the missing 11
        colour_list.append("darkblue") # for Robby
        
        return sns.heatmap(matrix, annot=True, square=True, cbar=False, vmin=-3, vmax=12, center= 5, cmap= colour_list, xticklabels="", yticklabels="")


individual = [6, 6, 5, 1, 0, 1, 4, 1, 5, 3, 3, 5, 1, 2, 5, 1, 2, 6, 3, 3, 5, 0, 0, 1, 2, 0, 0, 3, 4, 5, 0, 0, 5, 2, 5, 0, 2, 6, 0, 0, 4, 5, 2, 4, 5, 3, 6, 5, 0, 1, 1, 2, 6, 5, 0, 1, 0, 5, 0, 3, 0, 0, 5, 0, 2, 4, 3, 5, 4, 3, 2, 2, 6, 1, 3, 2, 5, 5, 3, 1, 5, 3, 4, 5, 1, 1, 5, 1, 4, 5, 3, 5, 5, 1, 2, 5, 1, 0, 5, 3, 2, 5, 0, 1, 1, 6, 2, 5, 3, 1, 5, 2, 6, 5, 2, 4, 5, 3, 0, 5, 2, 6, 5, 2, 4, 0, 6, 3, 5, 0, 3, 5, 4, 1, 5, 0, 2, 5, 0, 5, 2, 4, 1, 5, 0, 4, 5, 0, 6, 2, 2, 1, 0, 2, 2, 5, 6, 1, 6, 1, 1, 5, 1, 5, 2, 5, 0, 3, 4, 5, 5, 1, 4, 5, 1, 0, 1, 1, 2, 5, 3, 5, 3, 5, 6, 5, 4, 1, 5, 1, 4, 2, 5, 2, 5, 0, 3, 5, 3, 3, 5, 3, 5, 4, 5, 3, 5, 3, 4, 5, 3, 2, 3, 5, 3, 6, 5, 1, 6, 0, 6, 5, 2, 0, 5, 0, 1, 4, 4, 0, 4, 5, 5, 5, 3, 0, 2, 1, 3, 0, 4, 1, 2, 1, 2, 0, 6, 0, 2, 6, 4, 6, 2, 0]
 
environment = Environment()

map = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [-1,  8, 0,  2,  2,  9,  6,  0,  3,  5,  4, -1],
 [-1,  9,  9, 10,  8,  2,  7,  8,  1,  5,  0, -1],
 [-1,  5,  6,  6,  9,  5,  6,  6,  3,  5,  5, -1],
 [-1,  4,  9, 10,  9,  2, 10,  7,  8,  4,  2, -1],
 [-1,  5,  5,  2, 10, 0,  5,  2,  5, 10,  9, -1],
 [-1,  8,  1,  3,  6,  7,  0,  7,  1,  4,  9, -1],
 [-1,  2, 10,  1,  5, 10,  3, 10,  9,  8, 10, -1],
 [-1,  1,  7,  0, 10,  9,  4,  9,  4,  9,  3, -1],
 [-1,  3,  8, 1,  3,  3,  4, 10,  3,  5, 10, -1],
 [-1,  4,  2,  4,  5,  5,  0,  5,  1,  8,  8, -1],
 [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]


environment.__setMatrix__(map)
robby_row = 3
robby_column = 1

print(individual[-11:])

# Calculates state of a quarter.
def calc_state_one_quarter(environment, row, column, base, individual):
    if ((row <= 0) or (row >= NUM_ENVIRONMENT_ROWS+1)) or (
            (column <= 0) or (column >= NUM_ENVIRONMENT_COLUMNS+1)):
        return base * WALL
    else:
        water_quantity = environment.__getValue__(row, column)
        # if the gene associated with the amount of water is even then Robby sees "red" otherwise it sees as a "green" zone
        if individual[WATER_COLOUR_BASE + water_quantity]%2 == 0:
            return base * RED
        return base * GREEN



# Performs an action and returns the reward for it.
def perform_action(action, environment_local, individual):
    global robby_row, robby_column, fitness_points_per_quantity, ims, environment, pick_up_rate, collected_water
    reward = 0

    #print(action)
    # NORTH
    n_row = robby_row - 1
    n_column = robby_column

    if action == MOVE_NORTH:
        if (((n_row <= 0) or (n_row >= NUM_ENVIRONMENT_ROWS+1))
                or ((n_column <= 0) or (n_column >= NUM_ENVIRONMENT_COLUMNS+1))):
            reward = WALL_PENALTY
        else:
            robby_row = robby_row - 1

    # SOUTH
    n_row = robby_row + 1
    n_column = robby_column

    if action == MOVE_SOUTH:
        if (((n_row <= 0) or (n_row >= NUM_ENVIRONMENT_ROWS+1))
                or ((n_column <= 0) or (n_column >= NUM_ENVIRONMENT_COLUMNS+1))):
            reward = WALL_PENALTY
        else:
            robby_row = robby_row + 1

    # EAST
    n_row = robby_row
    n_column = robby_column + 1

    if action == MOVE_EAST:
        if (((n_row <= 0) or (n_row >= NUM_ENVIRONMENT_ROWS+1))
                or ((n_column <= 0) or (n_column >= NUM_ENVIRONMENT_COLUMNS+1))):
            reward = WALL_PENALTY
        else:
            robby_column = robby_column + 1

    # WEST
    n_row = robby_row
    n_column = robby_column - 1

    if action == MOVE_WEST:
        if (((n_row <= 0) or (n_row >= NUM_ENVIRONMENT_ROWS+1))
                or ((n_column <= 0) or (n_column >= NUM_ENVIRONMENT_COLUMNS+1))):
            reward = WALL_PENALTY
        else:
            robby_column = robby_column - 1

    if action == STAY_PUT:
        None  # do nothing

    if action == PICK_UP:
        pick_up_rate += 1
        print("Pick up --> ", robby_row, "  ", robby_column)
        water_quantity = environment.__getValue__(robby_row, robby_column)
        collected_water += fitness_points_per_quantity[water_quantity]
        environment.__setValue__(robby_row, robby_column, MAP_TYPE)


    if action == RANDOM_MOVE:
        random_step = np.random.randint(low=0, high=4)
        # steps randomly --> 0,1,2,3
        reward = perform_action(random_step, environment, individual)


    return reward

pick_up_rate = 0
collected_water = 0
fig = plt.figure()
plt.axes()
loop_count = 0
ani = 0
e = environment
ax = sns.heatmap(np.asarray(e.__getEnvironmentMatrix__()), annot=True, square=True, cbar=False, vmin=-2, vmax=12, center= 5, cmap= 'Reds', xticklabels="", yticklabels="")

# Initializes the plot.
def init():
    plt.clf()
    ax = sns.heatmap(np.asarray(e.__getEnvironmentMatrix__()), annot=True, square=True, cbar=False, vmin=-2, vmax=12, center= 5, cmap= 'Reds', xticklabels="", yticklabels="")

# Sets up stage for our little creature.
def calc_fitness_one_environment(individual2):
    global robby_row, robby_column, individual, environment, individual, ani
    fitness_points = 0
    print("In calculate fitness function")
    robby_row = 5 # or randnr(1, 10)
    robby_column = 5 # or randnr(1, 10)

    ani = animation.FuncAnimation(fig, foraging, frames=200, interval=1000)
       

# Foraging function.
def foraging(i):
    global environment, robby_row, robby_column, individual, ax
    state = 0

    state += calc_state_one_quarter(environment=environment, row=robby_row, column=robby_column,
                                                base=CENTER_BASE, individual=individual)
    state += calc_state_one_quarter(environment=environment, row=robby_row - 1, column=robby_column,
                                                base=NORTH_BASE, individual=individual)
    state += calc_state_one_quarter(environment=environment, row=robby_row, column=robby_column + 1,
                                                base=EAST_BASE, individual=individual)
    state += calc_state_one_quarter(environment=environment, row=robby_row + 1, column=robby_column,
                                                base=SOUTH_BASE, individual=individual)
    state += calc_state_one_quarter(environment=environment, row=robby_row, column=robby_column - 1,
                                                base=WEST_BASE, individual=individual)

    action = individual[state-1]
        
    
    perform_action(action, environment, individual)
    plt.clf()
    ax = environment.__drawPerceptedFrame__(robby_row, robby_column, individual[-WATER_QUANTITIES:])
    
    return ax

 
calc_fitness_one_environment(individual)
writer = PillowWriter(fps = 5)
ani.save("fix_map_ga_ii_visualized.gif", writer=writer)
print(pick_up_rate)
print(collected_water)