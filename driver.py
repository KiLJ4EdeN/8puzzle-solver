# remove this for windows platforms (resource only works on unix)
import resource
import heapq
import sys
import numpy as np
from abc import ABC, abstractmethod
from collections import deque


# Board Representation


class Board(object):
    # obj globals
    origin = None
    state = None
    operation = None
    n = 0
    zero = None
    cost = 0

    def __init__(self, state, origin=None, operation=None, n=0):
        # the original board
        self.origin = origin
        # the state it is in now
        self.state = np.array(state)
        # the operation done last
        self.operation = operation
        # the depth of the search
        self.n = n
        # find the starting state
        self.zero = self.discover_start()
        # calculate the cost based on the current state and the goal defined + the current depth
        self.cost = self.n + self.calc_manhattan_dist()

    def __lt__(self, other):
        # function to compare costs
        if self.cost != other.cost:
            return self.cost < other.cost
        else:
            operat = {'Up': 0, 'Down': 1, 'Left': 2, 'Right': 3}
            return operat[self.operation] < operat[other.operation]

    def __str__(self):
        # str method to nicely display the board
        return str(self.state[:3]) + '\n' \
               + str(self.state[3:6]) + '\n' \
               + str(self.state[6:]) + ' ' + str(self.n) + str(self.operation) + '\n'

    def check_obj(self):
        # function to check if the objective is reached
        if np.array_equal(self.state, np.arange(9)):
            return True
        else:
            return False

    def discover_start(self):
        # state start
        for i in range(9):
            if self.state[i] == 0:
                return i

    def calc_manhattan_dist(self):
        # manhattan dist calculator
        state = self.calc_index(self.state)
        goal = self.calc_index(np.arange(9))
        return sum((abs(state // 3 - goal // 3) + abs(state % 3 - goal % 3))[1:])

    @staticmethod
    def calc_index(state):
        indexes = np.array(range(9))
        for x, y in enumerate(state):
            indexes[y] = x
        return indexes

    def change_state(self, i, j):
        changed_state = np.array(self.state)
        changed_state[i], changed_state[j] = changed_state[j], changed_state[i]
        return changed_state

    # define how the moves affect the blocks.

    def move_up(self):
        if self.zero > 2:
            return Board(self.change_state(self.zero, self.zero - 3), self, 'Up', self.n + 1)
        else:
            return None

    def move_down(self):
        if self.zero < 6:
            return Board(self.change_state(self.zero, self.zero + 3), self, 'Down', self.n + 1)
        else:
            return None

    def move_left(self):
        if self.zero % 3 != 0:
            return Board(self.change_state(self.zero, self.zero - 1), self, 'Left', self.n + 1)
        else:
            return None

    def move_right(self):
        if (self.zero + 1) % 3 != 0:
            return Board(self.change_state(self.zero, self.zero + 1), self, 'Right', self.n + 1)
        else:
            return None

    # get the neighbors given the current state
    def neighbors(self):
        neighbors = [self.move_up(), self.move_down(), self.move_left(), self.move_right()]
        return list(filter(None, neighbors))

    __repr__ = __str__


# Create A base class for Search methods


class Search(ABC):
    solution = None
    frontier = None
    nodes_expanded = 0
    max_n = 0
    explored_nodes = set()
    initial_state = None

    def __init__(self, initial_state):
        self.initial_state = initial_state

    def parent_node(self):
        current = self.solution
        chain = [current]
        while current.origin is not None:
            chain.append(current.origin)
            current = current.origin
        return chain

    @property
    def path(self):
        path = [node.operation for node in self.parent_node()[-2::-1]]
        return path

    @abstractmethod
    def search(self):
        pass

    def set_solution(self, board):
        self.solution = board
        self.nodes_expanded = len(self.explored_nodes) - len(self.frontier) - 1
        # return self.solution


# Search Methods


class BreadthFirstSearch(Search):
    def __init__(self, initial_state):
        super(BreadthFirstSearch, self).__init__(initial_state)
        self.frontier = deque()

    def search(self):
        self.frontier.append(self.initial_state)
        while self.frontier:
            board = self.frontier.popleft()
            self.explored_nodes.add(tuple(board.state))
            if board.check_obj():
                self.set_solution(board)
                break
            # breadth search
            for neighbor in board.neighbors():
                if tuple(neighbor.state) not in self.explored_nodes:
                    self.frontier.append(neighbor)
                    self.explored_nodes.add(tuple(neighbor.state))
                    self.max_n = max(self.max_n, neighbor.n)
        return


class DepthFirstSearch(Search):
    def __init__(self, initial_state):
        super(DepthFirstSearch, self).__init__(initial_state)
        self.frontier = []

    def search(self):
        self.frontier.append(self.initial_state)
        while self.frontier:
            board = self.frontier.pop()
            # add the new tried states to the explored nodes
            self.explored_nodes.add(tuple(board.state))
            # if the explored node matches the solution stop the algorithm
            if board.check_obj():
                self.set_solution(board)
                break
            # depth search
            for neighbor in board.neighbors()[::-1]:
                if tuple(neighbor.state) not in self.explored_nodes:
                    self.frontier.append(neighbor)
                    self.explored_nodes.add(tuple(neighbor.state))
                    self.max_n = max(self.max_n, neighbor.n)
        return


class AStarSearch(Search):
    def __init__(self, initial_state):
        super(AStarSearch, self).__init__(initial_state)
        self.frontier = []

    def search(self):
        heapq.heappush(self.frontier, self.initial_state)
        # start with the initial state
        while self.frontier:
            board = heapq.heappop(self.frontier)
            # add the new tried states to the explored nodes
            self.explored_nodes.add(tuple(board.state))
            # if the explored node matches the solution stop the algorithm
            if board.check_obj():
                self.set_solution(board)
                break
            for neighbor in board.neighbors():
                if tuple(neighbor.state) not in self.explored_nodes:
                    heapq.heappush(self.frontier, neighbor)
                    self.explored_nodes.add(tuple(neighbor.state))
                    self.max_n = max(self.max_n, neighbor.n)
        return


def driver():
    problem = Board(np.array(eval(sys.argv[2])))
    method = sys.argv[1]
    if 'bfs' in method:
        solver = BreadthFirstSearch(problem)
    elif 'dfs' in method:
        solver = DepthFirstSearch(problem)
    elif 'ast' in method:
        solver = AStarSearch(problem)
    else:
        print("Invalid Arguments, call the application like this:\npython3 driver.py <method> <board>")
        solver = None
        exit(0)
    solver.search()

    file = open(f'output.txt', 'w')
    # paths we went to find the goal
    file.write('path_to_goal: ' + str(solver.path) + '\n')
    # how many pathing were used
    file.write('cost_of_path: ' + str(len(solver.path)) + '\n')
    file.write('nodes_expanded: ' + str(solver.nodes_expanded) + '\n')
    file.write('nodes_explored: ' + str(len(solver.explored_nodes)) + '\n')
    # how much the search went on
    file.write('search_depth: ' + str(solver.solution.n) + '\n')
    # maximum searching depth
    file.write('max_search_depth: ' + str(solver.max_n) + '\n')
    # resources used
    file.write('running_time: ' + str(resource.getrusage(resource.RUSAGE_SELF).ru_utime + \
                                      resource.getrusage(resource.RUSAGE_SELF).ru_stime) + '\n')
    file.write('max_ram_usage: ' + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    file.close()


if __name__ == "__main__":
    driver()
