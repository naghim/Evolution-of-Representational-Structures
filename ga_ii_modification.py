import array
import random
import sys

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

################################# PARAMETERS #################################

NUM_MOVES = 200
NUM_FORAGING = 100
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
SIGHT_PENALTY = -0.01

MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_EAST = 2
MOVE_WEST = 3
STAY_PUT = 4
PICK_UP = 5
RANDOM_MOVE = 6
ROTATE_LEFT = 7
ROTATE_RIGHT = 8

TOTAL_MOVES = 8
POSSIBLE_MOVE_VARIATIONS = 243
WATER_COLOUR_BASE = 243

STARTING_COORDINATES = 5

fitness_points_per_quantity = {
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

    def __getValue__(self, idx_row, idx_col):
        return self.rows[idx_row][idx_col]

    def __setitem__(self, idx_row, value):
        self.rows[idx_row] = value

    def __getitem__(self, idx_row):
        return self.rows[idx_row]

    def __printEnvironment__(self):
        print(np.matrix(self.rows))

    def __rotate_environment_right__(self, idx_row, idx_col):
        # a b c        g d a
        # d e f   -->  h e b
        # g h i        i f c
        
        moved = False
        N = self.n
        mat = self.rows
        for x in range(0, int(N/2)): 
          for y in range(x, N-x-1): 
                
              if idx_row == x and idx_col == y and moved==False:
                  idx_row = y
                  idx_col = N-1-x
                  moved = True

              # top-left
              temp = self.rows[x][y] 

              # top-left = bottom-left
              self.rows[x][y] = self.rows[N-1-y][x] 

              # bottom-left = bottom-right
              self.rows[N-1-y][x] = self.rows[N-1-x][N-1-y] 

              # bottom-right = top-right 
              self.rows[N-1-x][N-1-y] = self.rows[y][N-1-x] 

              # top-right = top-left
              self.rows[y][N-1-x] = temp

        #print(np.matrix(self.rows))
        return (idx_row, idx_col)

    def __rotate_environment_left__(self, idx_row, idx_col):
        # a b c        c f i
        # d e f   -->  b e h
        # g h i        a d g
        
        moved = False
        N = self.n
        mat = self.rows
        for x in range(0, int(N/2)): 
          for y in range(x, N-x-1): 
                
              if idx_row == x and idx_col == y and moved==False:
                  idx_row = N-1-y
                  idx_col = x
                  moved = True

              # top-left
              temp = self.rows[x][y] 
  
              # top-left = top-right  
              self.rows[x][y] = self.rows[y][N-1-x] 
    
              # top-right = bottom-right
              self.rows[y][N-1-x] = self.rows[N-1-x][N-1-y] 
    
              # bottom-right = bottom-left
              self.rows[N-1-x][N-1-y] = self.rows[N-1-y][x] 
    
              # bottom-left = top-left
              self.rows[N-1-y][x] = temp 
        #print(np.matrix(self.rows))
        return (idx_row, idx_col)


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
def perform_action(action):
    global robby_row, robby_column, fitness_points_per_quantity, environment
    reward = 0

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
        water_quantity = environment.__getValue__(robby_row, robby_column)
        reward = fitness_points_per_quantity[water_quantity]
        environment.__setValue__(robby_row, robby_column, 0)

    if action == RANDOM_MOVE:
        random_step = np.random.randint(low=0, high=4)
        # steps randomly --> 0,1,2,3
        reward = perform_action(random_step)

    if action == ROTATE_LEFT:
        robby_row, robby_column = environment.__rotate_environment_right__(robby_row, robby_column)

    if action == ROTATE_RIGHT:
        robby_row, robby_column = environment.__rotate_environment_left__(robby_row, robby_column)

    return reward
    

# Calculates the overall fitness of an individual.
# It forages NUM_FORAGING times and in each session
# it can perform an action NUM_MOVES times.
def calc_fitness_one_environment(individual):
    global robby_row, robby_column, environment
    fitness_points = 0

    for foraging_number in range(1, NUM_FORAGING + 1):
        robby_row = 5
        robby_column = 5
        environment = Environment()
        for move_number in range(1, NUM_MOVES + 1):
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
            fitness_points = fitness_points + perform_action(action)

    return fitness_points / NUM_FORAGING,



toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMax)
toolbox.register("attributes", random.randint, 0, 8)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attributes, 243 + 11)


# main function
def main():
    global toolbox

    pool = multiprocessing.Pool(processes=32)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=8, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", calc_fitness_one_environment)
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
