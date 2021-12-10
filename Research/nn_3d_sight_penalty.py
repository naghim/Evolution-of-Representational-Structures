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

DOWN_DOWN_BASE = 531441  # 3^12
UP_UP_BASE = 177147  # 3^11
NORTH_NORTH_BASE = 59049  # 3^10
SOUTH_SOUTH_BASE = 19683  # 3^9
EAST_EAST_BASE = 6561  # 3^8
WEST_WEST_BASE = 2187  # 3^7

DOWN_BASE = 729  # 3^6
UP_BASE = 243  # 3^5
NORTH_BASE = 81  # 3^4
SOUTH_BASE = 27  # 3^3
EAST_BASE = 9  # 3^2
WEST_BASE = 3  # 3^1
CENTER_BASE = 1  # 3^0

GREEN = 1  # analogously to CAN cell
RED = -1 #0  # analogously to EMPTY cell
WALL = 2

WALL_PENALTY = -5

### Commands/actions ###

MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_EAST = 2
MOVE_WEST = 3
MOVE_UP = 4
MOVE_DOWN = 5
PICK_UP = 6
RANDOM_MOVE = 7
ROTATE_LEFT = 8
ROTATE_RIGHT = 9
ROTATE_UP = 10
ROTATE_DOWN = 11
ROTATE_FORWARD = 12
ROTATE_BACKWARD = 13
MEMORY_1 = 14
MEMORY_2 = 15
MEMORY_3 = 16
MEMORY_4 = 17

WEIGHTS = 1128

WATER_COLOUR_BASE = WEIGHTS
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

robby_row = robby_column = STARTING_COORDINATES

### Input layer node indexes ###
BIAS = 0
CENTER = 1
NORTH = 2
NORTH2 = 3
EAST = 4
EAST2 = 5
SOUTH = 6
SOUTH2 = 7
WEST = 8
WEST2 = 9
UP = 10
UP2 = 11
DOWN = 12
DOWN2 = 13
MEMORY1 = 14
MEMORY2 = 15
MEMORY3 = 16
MEMORY4 = 17
AGE = 18

HIDDEN_LAYER_NEURONS = 30
INPUT_LAYER_NEURONS = 19
OUTPUT_LAYER_NEURONS = 18
OFFSET_INPUT_LAYER_WEIGHTS = 0
OFFSET_HIDDEN_LAYER_WEIGHTS = HIDDEN_LAYER_NEURONS * INPUT_LAYER_NEURONS

### Sight ###
SIGHT_DIRECTION_BASE = WATER_COLOUR_BASE + 10
SIGHT_CURRENT = SIGHT_DIRECTION_BASE + 1
SIGHT_EAST = SIGHT_DIRECTION_BASE + 2
SIGHT_SOUTH = SIGHT_DIRECTION_BASE + 3
SIGHT_WEST = SIGHT_DIRECTION_BASE + 4
SIGHT_NORTH = SIGHT_DIRECTION_BASE + 5
SIGHT_UP = SIGHT_DIRECTION_BASE + 6
SIGHT_DOWN = SIGHT_DIRECTION_BASE + 7

SIGHT_EAST_EAST = SIGHT_DIRECTION_BASE + 8
SIGHT_SOUTH_SOUTH = SIGHT_DIRECTION_BASE + 9
SIGHT_WEST_WEST = SIGHT_DIRECTION_BASE + 10
SIGHT_NORTH_NORTH = SIGHT_DIRECTION_BASE + 11
SIGHT_UP_UP = SIGHT_DIRECTION_BASE + 12
SIGHT_DOWN_DOWN = SIGHT_DIRECTION_BASE + 13

SIGHT_PENALTY = -0.005


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

    def __rotate_environment_right__(self, idx_row, idx_col):
        # a b c        g d a
        # d e f   -->  h e b
        # g h i        i f c

        moved = False
        N = self.n
        for z in range(1, self.s):
            mat = self.cube[z]
            for x in range(0, int(N / 2)):
                for y in range(x, N - x - 1):

                    if idx_row == x and idx_col == y and moved == False:
                        idx_row = y
                        idx_col = N - 1 - x
                        moved = True

                    # top-left
                    temp = mat[x][y]

                    # top-left = bottom-left
                    mat[x][y] = mat[N - 1 - y][x]

                    # bottom-left = bottom-right
                    mat[N - 1 - y][x] = mat[N - 1 - x][N - 1 - y]

                    # bottom-right = top-right
                    mat[N - 1 - x][N - 1 - y] = mat[y][N - 1 - x]

                    # top-right = top-left
                    mat[y][N - 1 - x] = temp

        # print(np.matrix(self.cube[1]))
        return (idx_row, idx_col)

    def __rotate_environment_left__(self, idx_row, idx_col):
        # a b c        c f i
        # d e f   -->  b e h
        # g h i        a d g

        moved = False
        N = self.n
        for z in range(1, self.s):
            mat = self.cube[z]
            for x in range(0, int(N / 2)):
                for y in range(x, N - x - 1):

                    if idx_row == x and idx_col == y and moved == False:
                        idx_row = N - 1 - y
                        idx_col = x
                        moved = True

                    # top-left
                    temp = mat[x][y]

                    # top-left = top-right
                    mat[x][y] = mat[y][N - 1 - x]

                    # top-right = bottom-right
                    mat[y][N - 1 - x] = mat[N - 1 - x][N - 1 - y]

                    # bottom-right = bottom-left
                    mat[N - 1 - x][N - 1 - y] = mat[N - 1 - y][x]

                    # bottom-left = top-left
                    mat[N - 1 - y][x] = temp
                    # print(np.matrix(self.cube[1]))
        return (idx_row, idx_col)

    def rotate_up(self, idx_slice, idx_row, idx_column):
        new_cube = np.zeros((self.s, self.n, self.m), dtype=int)
        for z in range(0, self.s):
            for x in range(0, self.s):
                new_cube[z][x] = self.cube[x][NUM_ENVIRONMENT_ROWS - z + 1]
                # print(new_cube[z][x])
                # print(z,x, " == ", x, NUM_ENVIRONMENT_ROWS-z+1)
        self.cube = new_cube

        # returns the new position, the column index remains the same
        return NUM_ENVIRONMENT_ROWS - idx_row + 1, idx_slice, idx_column

    def rotate_down(self, idx_slice, idx_row, idx_column):
        new_cube = np.zeros((self.s, self.n, self.m), dtype=int)
        for z in range(0, self.s):
            for x in range(0, self.s):
                new_cube[z][x] = self.cube[NUM_ENVIRONMENT_SLICES - x + 1][z]
                # print(new_cube[z][x])
                # print(z,x, " == ", NUM_ENVIRONMENT_ROWS-x+1,z)
        self.cube = new_cube

        # returns the new position, the column index remains the same
        return idx_slice, NUM_ENVIRONMENT_SLICES - idx_row + 1, idx_column


# Calculates the state of a cube cell.
def calculate_state_one_cell(cubeslice, row, column, base, individual):
    global environment

    if ((row <= 0) or (row >= (NUM_ENVIRONMENT_ROWS + 1))) or (
            (column <= 0) or (column >= (NUM_ENVIRONMENT_COLUMNS + 1))) or (
            (cubeslice <= 0) or (cubeslice >= NUM_ENVIRONMENT_SLICES)):
        return base * WALL
    else:
        water_quantity = environment[cubeslice][row][column]
        # if the gene associated with the amount of water is greater than 0 then Robby sees "greed" otherwise it sees it as a "red" zone
        if individual[WATER_COLOUR_BASE + water_quantity] > 0:
            return base * GREEN
        return base * RED


# Performs an action and returns the reward for it.
def perform_action(action, individual):
    global robby_cubeslice, robby_row, robby_column, FITNESS_POINTS_PER_QUANTITY, environment, MEMORY_NEURON_1, MEMORY_NEURON_2, MEMORY_NEURON_3, MEMORY_NEURON_4
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

        return reward

    # SOUTH
    n_row = robby_row + 1
    n_column = robby_column

    if action == MOVE_SOUTH:
        if ((n_row <= 0) or (n_row >= (NUM_ENVIRONMENT_ROWS + 1))) or (
                (n_column <= 0) or (n_column >= (NUM_ENVIRONMENT_COLUMNS + 1))) or (
                (robby_cubeslice <= 0) or (robby_cubeslice >= NUM_ENVIRONMENT_SLICES)):
            reward = WALL_PENALTY
        else:
            robby_row = robby_row + 1

        return reward

    # EAST
    n_row = robby_row
    n_column = robby_column + 1

    if action == MOVE_EAST:
        if ((n_row <= 0) or (n_row >= (NUM_ENVIRONMENT_ROWS + 1))) or (
                (n_column <= 0) or (n_column >= (NUM_ENVIRONMENT_COLUMNS + 1))) or (
                (robby_cubeslice <= 0) or (robby_cubeslice >= NUM_ENVIRONMENT_SLICES)):
            reward = WALL_PENALTY
        else:
            robby_column = robby_column + 1

        return reward

    # WEST
    n_row = robby_row
    n_column = robby_column - 1

    if action == MOVE_WEST:
        if ((n_row <= 0) or (n_row >= (NUM_ENVIRONMENT_ROWS + 1))) or (
                (n_column <= 0) or (n_column >= (NUM_ENVIRONMENT_COLUMNS + 1))) or (
                (robby_cubeslice <= 0) or (robby_cubeslice >= NUM_ENVIRONMENT_SLICES)):
            reward = WALL_PENALTY
        else:
            robby_column = robby_column - 1

        return reward

    if action == PICK_UP:
        water_quantity = environment[robby_cubeslice][robby_row][robby_column]
        reward = FITNESS_POINTS_PER_QUANTITY[water_quantity]
        environment[robby_cubeslice][robby_row][robby_column] = 0

        return reward

    if action == RANDOM_MOVE:
        random_step = np.random.randint(low=0, high=6)
        # steps randomly --> 0,1,2,3  # 4 up, 5 down
        reward = perform_action(random_step, individual)

        return reward

    ### 3D moveset

    # UP
    if action == MOVE_UP:
        n_cubeslice = robby_cubeslice + 1
        if ((robby_row <= 0) or (robby_row >= (NUM_ENVIRONMENT_ROWS + 1))) or (
                (robby_column <= 0) or (robby_column >= (NUM_ENVIRONMENT_COLUMNS + 1))) or (
                (n_cubeslice <= 0) or (n_cubeslice >= NUM_ENVIRONMENT_SLICES)):
            reward = WALL_PENALTY
        else:
            robby_cubeslice += 1

        return reward

    # DOWN
    if action == MOVE_DOWN:
        n_cubeslice = robby_cubeslice - 1
        if ((robby_row <= 0) or (robby_row >= (NUM_ENVIRONMENT_ROWS + 1))) or (
                (robby_column <= 0) or (robby_column >= (NUM_ENVIRONMENT_COLUMNS + 1))) or (
                (n_cubeslice <= 0) or (n_cubeslice >= NUM_ENVIRONMENT_SLICES)):
            reward = WALL_PENALTY
        else:
            robby_cubeslice -= 1

        return reward

    # ROTATE_LEFT = 8
    if action == ROTATE_LEFT:
        robby_row, robby_column = environment.__rotate_environment_right__(robby_row, robby_column)
        return reward

    # ROTATE_RIGHT = 9
    if action == ROTATE_RIGHT:
        robby_row, robby_column = environment.__rotate_environment_left__(robby_row, robby_column)
        return reward

    # ROTATE_UP = 10
    if action == ROTATE_UP:
        robby_cubeslice, robby_row, robby_column = environment.rotate_up(robby_cubeslice, robby_row, robby_column)
        return reward

    # ROTATE_DOWN = 11
    if action == ROTATE_DOWN:
        robby_cubeslice, robby_row, robby_column = environment.rotate_down(robby_cubeslice, robby_row, robby_column)
        return reward

    # ROTATE_1 = 12
    if action == ROTATE_FORWARD:
        return reward

    # ROTATE_2 = 13
    if action == ROTATE_BACKWARD:
        robby_row, robby_column = environment.__rotate_environment_left__(robby_row, robby_column)
        robby_row, robby_column = environment.__rotate_environment_left__(robby_row, robby_column)
        return reward

    print("Error: action not recognized")
    return reward


class Neural_Network():
    def __init__(self, W1, W2):
        self.inputSize = INPUT_LAYER_NEURONS
        self.outputSize = OUTPUT_LAYER_NEURONS
        self.hiddenSize = HIDDEN_LAYER_NEURONS
        self.W1 = W1
        self.W2 = W2

    def forward(self, state):
        self.input_x_weights = np.dot(state, self.W1)
        self.hidden_layer_activation = self.relu(self.input_x_weights)
        self.hidden_layer_activation = np.append(self.hidden_layer_activation, 1)
        self.output_layer = np.dot(self.hidden_layer_activation, self.W2)

        action_neurons = self.output_layer[0:ROTATE_LEFT]

        winner = get_maximums_position(action_neurons)

        state[MEMORY_1] = self.output_layer[MEMORY1]
        state[MEMORY_2] = self.output_layer[MEMORY2]
        state[MEMORY_3] = self.output_layer[MEMORY3]
        state[MEMORY_4] = self.output_layer[MEMORY4]

        return winner, state

    def relu(self, vector):
        return relu_for_vector(vector)


# Helper function, returns first encountered max's position
def get_maximums_position(array):
    max_value = array[0]
    max_pos = 0

    for i in range(0, len(array)):
        if array[i] > max_value:
            max_value = array[i]
            max_pos = i

    return max_pos


# Helper function, returns a matrix made from a 1xn vector.
def vector_to_matrix(vector, dimensions, offset):
    matrix = np.zeros((dimensions[0], dimensions[1]))
    for i in range(0, dimensions[0]):
        from_idx = i * dimensions[1] + offset
        to_idx = from_idx + dimensions[1]
        matrix[i] = vector[from_idx: to_idx]  # [) intervall

    return matrix


# Relu for vector. Returns a new vector.
def relu_for_vector(vector):
    for i in range(0, len(vector)):
        element = vector[i]
        vector[i] = element if element > 0 else 0

    return vector


# Calculates the overall fitness of an individual.
# It forages NUM_FORAGING times and in each session
# it can perform an action NUM_MOVES times.
def calculate_individual_fitness(individual):
    global robby_cubeslice, robby_row, robby_column, environment, MEMORY_NEURON_1, MEMORY_NEURON_2, MEMORY_NEURON_3, MEMORY_NEURON_4
    fitness_points = 0

    weight1 = vector_to_matrix(individual, [INPUT_LAYER_NEURONS, HIDDEN_LAYER_NEURONS],
                               OFFSET_INPUT_LAYER_WEIGHTS)
    weight2 = vector_to_matrix(individual, [HIDDEN_LAYER_NEURONS + 1, OUTPUT_LAYER_NEURONS],
                               OFFSET_HIDDEN_LAYER_WEIGHTS)
    network = Neural_Network(weight1, weight2)

    state = np.zeros(19)
    state[BIAS] = 1

    for foraging_number in range(1, NUM_FORAGING + 1):
        robby_row = 5
        robby_column = 5
        robby_cubeslice = 5
        environment = Environment()

        for move_number in range(1, NUM_MOVES + 1):
            directions = 0


            if individual[SIGHT_CURRENT] >= 0:
                state[CENTER] = calculate_state_one_cell(cubeslice=robby_cubeslice, row=robby_row, column=robby_column,
                                                     base=1, individual=individual)
                directions += 1
            if individual[SIGHT_NORTH] >= 0:
                state[NORTH] = calculate_state_one_cell(cubeslice=robby_cubeslice, row=robby_row - 1, column=robby_column,
                                                    base=1, individual=individual)
                directions += 1
            if individual[SIGHT_EAST] >= 0:
                state[EAST] = calculate_state_one_cell(cubeslice=robby_cubeslice, row=robby_row, column=robby_column + 1,
                                                   base=1, individual=individual)
                directions += 1
            if individual[SIGHT_SOUTH] >= 0:
                state[SOUTH] = calculate_state_one_cell(cubeslice=robby_cubeslice, row=robby_row + 1, column=robby_column,
                                                    base=1, individual=individual)
                directions += 1
            if individual[SIGHT_WEST] >= 0:
                state[WEST] = calculate_state_one_cell(cubeslice=robby_cubeslice, row=robby_row, column=robby_column - 1,
                                                   base=1, individual=individual)
                directions += 1
            if individual[SIGHT_UP] >= 0:
                state[UP] = calculate_state_one_cell(cubeslice=robby_cubeslice + 1, row=robby_row, column=robby_column,
                                                 base=1, individual=individual)
                directions += 1
            if individual[SIGHT_DOWN] >= 0:
                state[DOWN] = calculate_state_one_cell(cubeslice=robby_cubeslice - 1, row=robby_row, column=robby_column,
                                                   base=1, individual=individual)
                directions += 1

            if individual[SIGHT_NORTH_NORTH] >= 0:
                state[NORTH2] = calculate_state_one_cell(cubeslice=robby_cubeslice, row=robby_row - 2, column=robby_column,
                                                    base=1, individual=individual)
                directions += 1
            if individual[SIGHT_EAST_EAST] >= 0:
                state[EAST2] = calculate_state_one_cell(cubeslice=robby_cubeslice, row=robby_row, column=robby_column + 2,
                                                    base=1, individual=individual)
                directions += 1
            if individual[SIGHT_SOUTH_SOUTH] >= 0:
                state[SOUTH2] = calculate_state_one_cell(cubeslice=robby_cubeslice, row=robby_row + 2, column=robby_column,
                                                    base=1, individual=individual)
                directions += 1
            if individual[SIGHT_WEST_WEST] >= 0:
                state[WEST2] = calculate_state_one_cell(cubeslice=robby_cubeslice, row=robby_row, column=robby_column - 2,
                                                    base=1, individual=individual)
                directions += 1
            if individual[SIGHT_UP_UP] >= 0:
                state[UP2] = calculate_state_one_cell(cubeslice=robby_cubeslice + 2, row=robby_row, column=robby_column,
                                                    base=1, individual=individual)
                directions += 1
            if individual[SIGHT_DOWN_DOWN] >= 0:
                state[DOWN2] = calculate_state_one_cell(cubeslice=robby_cubeslice - 2, row=robby_row, column=robby_column,
                                                    base=1, individual=individual)
                directions += 1

            state[AGE] = foraging_number

            action, state = network.forward(state)
            fitness_points = fitness_points + perform_action(action, individual) + (SIGHT_PENALTY * directions)

    return fitness_points / NUM_FORAGING,


toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)
toolbox.register("attributes", lambda: random.uniform(-1, 1))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attributes, WEIGHTS + 11 + 13) 


def main():
    global toolbox
    weights = 494

    pool = multiprocessing.Pool(processes=10)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", calculate_individual_fitness)
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    random.seed(170)
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 500, stats=stats, halloffame=hof, verbose=True)
    for h in hof:
        print(h)


if __name__ == '__main__':
    main()