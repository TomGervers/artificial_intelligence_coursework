# A* algorithm solution for a specific 8 puzzle problem using Manhattan
# distance as heuristic function

# Import necessary modules
from collections import defaultdict
from copy import deepcopy
import numpy as np
import time


# Calulcate the manhattan distance between misplaced tiles
def manhattan_distance(s, g):
    return sum((abs(s // 3 - g // 3) + abs(s % 3 - g % 3))[1:])


# Assigning each tile a coordinate
def coordinate(s):
    c = np.array(range(9))
    for x, y in enumerate(s):
        c[y] = x
    return c


# Solver function that uses the heuristic to solve the puzzle
def manhattanSolve(board, goal):
    # Possible moves
    moves = np.array(
        [
            ('u', [0, 1, 2], -3),
            ('d', [6, 7, 8], 3),
            ('l', [0, 3, 6], -1),
            ('r', [2, 5, 8], 1)
        ],
        dtype=[
            ('move', str, 1),
            ('pos', list),
            ('delta', int)
        ]
    )

    # State of puzzle
    STATE = [
        ('board', list),
        ('parent', int),
        ('gn', int),
        ('heuristicn', int)
    ]

    PRIORITY = [
        ('pos', int),
        ('fn', int)
    ]

    # Previous state of the board
    previous_boards = defaultdict(bool)

    # Coordinates of goal state
    goal_coordinate = coordinate(goal)

    # Calculate heuristic of n using manhattan distance function
    heuristicn = manhattan_distance(coordinate(board), goal_coordinate)

    state = np.array([(board, -1, 0, heuristicn)], STATE)

    priority = np.array([(0, heuristicn)], PRIORITY)

    # Loop through until solution found
    while True:
        priority = np.sort(priority, kind='mergesort', order=['fn', 'pos'])
        pos = priority[0][0]
        priority = np.delete(priority, 0, 0)
        board = state[pos][0]
        gn = state[pos][2] + 1
        loc = int(np.where(board == 0)[0])

        # Choose move
        for m in moves:
            if loc not in m['pos']:
                success = deepcopy(board)
                delta_loc = loc + m['delta']
                success[loc], success[delta_loc] = (
                    success[delta_loc], success[loc]
                )
                success_t = tuple(success)

                if previous_boards[success_t]:
                    continue

                previous_boards[success_t] = True

                hn = manhattan_distance(coordinate(success_t), goal_coordinate)
                state = np.append(
                    state,
                    np.array([(success, pos, gn, hn)], STATE),
                    0
                )
                priority = np.append(
                    priority,
                    np.array([(len(state) - 1, gn + hn)], PRIORITY),
                    0
                )

                # Check if equal to the goal state
                if np.array_equal(success, goal):
                    return state, len(priority)


# Generate the optimal solution
def generate_optimal(state):
    optimal = np.array([], int).reshape(-1, 9)
    last = len(state) - 1
    while last != -1:
        optimal = np.insert(optimal, 0, state[last]['board'], 0)
        last = int(state[last]['parent'])
    return optimal.reshape(-1, 3, 3)


# Main function
def main():
    # Define start and goal states
    start = np.array([3, 6, 4, 0, 1, 2, 8, 7, 5])
    goal = np.array([1, 2, 3, 8, 0, 4, 7, 6, 5])
    # Set the board to be the start state
    board = np.array(list(map(int, start)))

    # Start timer
    start = time.time()

    # Solve puzzle and find optimal
    state, explored = manhattanSolve(board, goal)
    optimal = generate_optimal(state)

    # Stop timer
    end = time.time()

    # Print path to solution
    print((
        '{}\n'
        '\n'
    ).format(optimal))

    # Print number of moves and execution time
    print((
        'Puzzle solved\n'
        'Number of moves in optimal path: {}\n'
    ).format(len(optimal) - 1))

    print('Execution time: ', end - start)


if __name__ == '__main__':
    main()
