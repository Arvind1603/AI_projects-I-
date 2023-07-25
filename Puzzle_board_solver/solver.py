

import sys
import puzz
import pdqpq


GOAL_STATE = puzz.EightPuzzleBoard("012345678")


def solve_puzzle(start_state, flavor):
    """Perform a search to find a solution to a puzzle.
    
    Args:
        start_state (EightPuzzleBoard): the start state for the search
        flavor (str): tag that indicate which type of search to run.  Can be one of the following:
            'bfs' - breadth-first search
            'ucost' - uniform-cost search
            'greedy-h1' - Greedy best-first search using a misplaced tile count heuristic
            'greedy-h2' - Greedy best-first search using a Manhattan distance heuristic
            'greedy-h3' - Greedy best-first search using a weighted Manhattan distance heuristic
            'astar-h1' - A* search using a misplaced tile count heuristic
            'astar-h2' - A* search using a Manhattan distance heuristic
            'astar-h3' - A* search using a weighted Manhattan distance heuristic
    
    Returns: 
        A dictionary containing describing the search performed, containing the following entries:
        'path' - list of 2-tuples representing the path from the start to the goal state (both 
            included).  Each entry is a (str, EightPuzzleBoard) pair indicating the move and 
            resulting successor state for each action.  Omitted if the search fails.
        'path_cost' - the total cost of the path, taking into account the costs associated with 
            each state transition.  Omitted if the search fails.
        'frontier_count' - the number of unique states added to the search frontier at any point 
            during the search.
        'expanded_count' - the number of unique states removed from the frontier and expanded 
            (successors generated)

    """
    if flavor.find('-') > -1:
        strat, heur = flavor.split('-')
    else:
        strat, heur = flavor, None

    if strat == 'bfs':
        return BreadthFirstSolver(GOAL_STATE).solve(start_state)
    elif strat == 'ucost':
        return UniformCostSolver(GOAL_STATE).solve(start_state)
    elif strat == 'greedy':
        return GreedySolver(GOAL_STATE).solve(start_state, heur)
    elif strat == 'astar':
        return AStarSolver(GOAL_STATE).solve(start_state, heur)
    else:
        raise ValueError("Unknown search flavor '{}'".format(flavor))


def print_table(flav__results, include_path=False):
    """Print out a comparison of search strategy results.

    Args:
        flav__results (dictionary): a dictionary mapping search flavor tags result statistics. See
            solve_puzzle() for detail.
        include_path (bool): indicates whether to include the actual solution paths in the table

    """
    result_tups = sorted(flav__results.items())
    c = len(result_tups)
    na = "{:>12}".format("n/a")
    rows = [  # abandon all hope ye who try to modify the table formatting code...
        "flavor  " + "".join([ "{:>12}".format(tag) for tag, _ in result_tups]),
        "--------" + ("  " + "-"*10)*c,
        "length  " + "".join([ "{:>12}".format(len(res['path'])) if 'path' in res else na 
                                for _, res in result_tups ]),
        "cost    " + "".join([ "{:>12,}".format(res['path_cost']) if 'path_cost' in res else na 
                                for _, res in result_tups ]),
        "frontier" + ("{:>12,}" * c).format(*[res['frontier_count'] for _, res in result_tups]),
        "expanded" + ("{:>12,}" * c).format(*[res['expanded_count'] for _, res in result_tups])
    ]
    if include_path:
        rows.append("path")
        longest_path = max([ len(res['path']) for _, res in result_tups if 'path' in res ] + [0])
        print("longest", longest_path)
        for i in range(longest_path):
            row = "        "
            for _, res in result_tups:
                if len(res.get('path', [])) > i:
                    move, state = res['path'][i]
                    row += " " + move[0] + " " + str(state)
                else:
                    row += " "*12
            rows.append(row)
    print("\n" + "\n".join(rows), "\n")


#fFor misplaced tiles
def h1(state):
    misplaced = 0
    for i in range(9):
        if GOAL_STATE.find(str(i)) != state.find(str(i)):
            misplaced += 1
    return misplaced

#manhattan distance
def h2(state):
    dist = 0
    for i in range(1, 9):
        temp1 = state.find(str(i))
        temp2 = GOAL_STATE.find(str(i))
        dist += (abs(temp1[0]-temp2[0]) + abs(temp1[1] - temp2[1]))
    return dist

#weighted manhattan distance
def h3(state):
    dist = 0
    for i in range(1, 9):
        temp1 = state.find(str(i))
        temp2 = GOAL_STATE.find(str(i))
        dist += ((abs(temp1[0]-temp2[0]) + abs(temp1[1] - temp2[1])) * (i**2))
    return dist


class inheritance:  # Setu up methods for all solver.

    def __init__(self, solve_state):
        self.parents = {}
        self.expanded_count = 0
        self.frontier_count = 0
        self.goal = solve_state

    def get_path(self, state):
        """Return the solution path from the start state of the search to a target.
        
        Results are obtained by retracing the path backwards through the parent tree to the start
        state for the search at the root.
        
        Args:
            state (EightPuzzleBoard): target state in the search tree
        
        Returns:
            A list of EightPuzzleBoard objects representing the path from the start state to the
            target state
        """
        path = []
        while state is not None:
            path.append(state)
            state = self.parents[state]
        path.reverse()
        return path

    def get_cost(self, state): 
        """Calculate the path cost from start state to a target state.
        
        Transition costs between states are equal to the square of the number on the tile that 
        was moved. 
        Args:
            state (EightPuzzleBoard): target state in the search tree
        
        Returns:
            Integer indicating the cost of the solution path
        """
        cost = 0
        path = self.get_path(state)
        for i in range(1, len(path)):
            x, y = path[i-1].find(None)  # the most recently moved tile leaves the blank behind
            tile = path[i].get_tile(x, y)        
            cost += int(tile)**2
        return cost

    def get_results_dict(self, state):
        """Construct the output dictionary for solve_puzzle()
        
        Args:
            state (EightPuzzleBoard): final state in the search tree
        
        Returns:
            A dictionary describing the search performed (see solve_puzzle())
        """
        results = {}
        results['frontier_count'] = self.frontier_count
        results['expanded_count'] = self.expanded_count
        if state:
            results['path_cost'] = self.get_cost(state)
            path = self.get_path(state)
            moves = ['start'] + [ path[i-1].get_move(path[i]) for i in range(1, len(path)) ]
            results['path'] = list(zip(moves, path))
        return results     

class BreadthFirstSolver(inheritance):

    def __init__(self, solve_state):
        self.frontier = pdqpq.FifoQueue()
        self.explored = set()
        super().__init__(solve_state)

    def add_to_frontier(self, node):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def solve(self, start_state):
        self.parents[start_state] = None
        self.add_to_frontier(start_state)

        if start_state == self.goal:  #if solved
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  
            succs = self.expand_node(node)

            for move, succ in succs.items():
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    # check for goal state before adding to frontier
                    if succ == self.goal:
                        return self.get_results_dict(succ)
                    else:
                        self.add_to_frontier(succ)
        # Something went wrong
        return self.get_results_dict(None) 


class UniformCostSolver(inheritance):

    def __init__(self, solve_state):
        self.frontier = pdqpq.PriorityQueue()
        self.explored = set()
        super().__init__(solve_state)

    def add_to_frontier(self, node, priortity):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node, priortity)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def solve(self, start_state):
        self.parents[start_state] = None
        self.add_to_frontier(start_state, 0)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            
            if node == self.goal:
                return self.get_results_dict(node)

            succs = self.expand_node(node)
            for move, succ in succs.items():
                # blank tile location
                x, y = node.find(None)
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    #cost from start state
                    priority = self.get_cost(node) + (int(succ.get_tile(x, y))**2)
                    self.add_to_frontier(succ, priority)

                elif ((succ in self.frontier) and ((self.get_cost(node) + int(succ.get_tile(x, y))**2)) < self.frontier.get(succ)):
                    self.parents[succ] = node
                    #update task in pqueue if lower cost found
                    self.frontier.add(succ)
        # Something went wrong
        return self.get_results_dict(None) 


class GreedySolver(inheritance):

    def __init__(self, solve_state):
        self.frontier = pdqpq.PriorityQueue()
        self.explored = set()
        super().__init__(solve_state)

    def add_to_frontier(self, node, priortity):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node, priortity)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def solve(self, start_state, heur):
        self.parents[start_state] = None
        self.add_to_frontier(start_state, 0)

        if start_state == self.goal:  #if Solved   
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            
            if node == self.goal:
                return self.get_results_dict(node)

            succs = self.expand_node(node)
            for move, succ in succs.items():
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    #call the heur functions
                    if heur == "h1": # misplaced
                        priority = h1(succ)
                    elif heur == "h2": # manhattan dist
                        priority = h2(succ)
                    elif heur == "h3": # weighted Manhattan distance
                        priority = h3(succ)
                    self.add_to_frontier(succ, priority)

                elif (succ in self.frontier) and (self.get_cost(succ) < self.frontier.get(succ)):
                    self.parents[succ] = node
                    #update task in pqueue if lower cost found
                    self.frontier.add(succ)

       # Something went wrong
        return self.get_results_dict(None)


class AStarSolver(inheritance):

    def __init__(self, solve_state):
        self.frontier = pdqpq.PriorityQueue()
        self.explored = set()
        super().__init__(solve_state)


    def add_to_frontier(self, node, priortity):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node, priortity)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def solve(self, start_state, heur):
        self.parents[start_state] = None
        self.add_to_frontier(start_state, 0)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            
            if node == self.goal:
                return self.get_results_dict(node)

            succs = self.expand_node(node)
            for move, succ in succs.items():
                x, y = node.find(None)
                if heur == "h1": # misplaced
                    priority = h1(succ)
                elif heur == "h2": # manhattan dist
                    priority = h2(succ)
                elif heur == "h3": # weighted Manhattan distance
                    priority = h3(succ)
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    priority += (self.get_cost(node) + int(succ.get_tile(x, y))**2)
                    self.add_to_frontier(succ, priority)

                elif (succ in self.frontier) and ((priority + self.get_cost(node) + int(succ.get_tile(x, y))**2) < self.frontier.get(succ)):
                    self.parents[succ] = node
                    #update task in pqueue if lower cost found
                    self.frontier.add(succ)
        # Something went wrong
        return self.get_results_dict(None)

def get_test_puzzles():
    """Return sample start states for testing the search strategies.
    
    Returns:
        A tuple containing three EightPuzzleBoard objects representing start states that have an
        optimal solution path length of 3-5, 10-15, and >=25 respectively.
    
    """

    x = puzz.EightPuzzleBoard("120345678")
    y = puzz.EightPuzzleBoard("456781230")
    z = puzz.EightPuzzleBoard("768012345")
    return (x, y, z)  # fix this line!

############################################

if __name__ == '__main__':

    # parse the command line args
    start = puzz.EightPuzzleBoard(sys.argv[1])
    if sys.argv[2] == 'all':
        flavors = ['bfs', 'ucost', 'greedy-h1', 'greedy-h2', 
                   'greedy-h3', 'astar-h1', 'astar-h2', 'astar-h3']
    else:
        flavors = sys.argv[2:]

    # run the search(es)
    results = {}
    for flav in flavors:
        print("solving puzzle {} with {}".format(start, flav))
        results[flav] = solve_puzzle(start, flav)

    print_table(results, include_path=False)  # change to True to see the paths!

