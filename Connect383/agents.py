import random
import math
#


BOT_NAME =  "Beta"# INSERT NAME FOR YOUR BOT HERE OR IT WILL THROW AN EXCEPTION


class RandomAgent:
    """Agent that picks a random available move.  You should be able to beat it."""
  
    rseed = None  # change this to a value if you want consistent random choices

    def __init__(self):
        if self.rseed is None:
            self.rstate = None
        else:
            random.seed(self.rseed)
            self.rstate = random.getstate()

    def get_move(self, state):
        if self.rstate is not None:
            random.setstate(self.rstate)
        return random.choice(state.successors())


class HumanAgent:
    """Prompts user to supply a valid move.  Very slow and not always smart."""

    def get_move(self, state, depth=None):
        move__state = dict(state.successors())
        prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
        move = None
        while move not in move__state:
            try:
                move = int(input(prompt))
            except ValueError:
                continue
        return move, move__state[move]


class MinimaxAgent:
    """Artificially intelligent agent that uses minimax to optimally select the best move."""

    def get_move(self, state):
        """Select the best available move, based on minimax value."""
        nextp = state.next_player()
        best_util = -math.inf if nextp == 1 else math.inf
        best_move = None
        best_state = None

        for move, state in state.successors():
            util = self.minimax(state)
            if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
                best_util, best_move, best_state = util, move, state
        return best_move, best_state

    def minimax(self, state):
        """Determine the minimax utility value of the given state.

        Gets called by get_move() to determine the value of each successor state.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the exact minimax utility value of the state
        """
        if state.is_full():
            return state.utility()
       
       
        min_max =[]
        next_p = state.next_player()
       
        for move, state in state.successors():
            min_max.append(self.minimax(state))
        if next_p == 1:
            return max(min_max)
        else:
            return min(min_max)
        

class MinimaxLookaheadAgent(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move.
 
    Hint: Consider what you did for MinimaxAgent. What do you need to change to get what you want? 
    """


    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax_depth(self, state, depth):
        next_p = state.next_player()
        if state.is_full() or depth == self.depth_limit:
            return self.evaluation(state)
        if next_p == 1:
            v = math.inf
            holder = state.successors()
            for move in holder:
                v = min(v, MinimaxLookaheadAgent.minimax_depth(move[1], depth + 1))
            return v
        if next_p == -1:
            v = -math.inf
            holder = state.successors()
            for move in holder:
                v = max(v, MinimaxLookaheadAgent.minimax_depth(move[1], depth + 1))
            return v

    def minimax(self, state, depth_limit):
        """Determine the heuristically estimated minimax utility value of the given state.

        Gets called by get_move() to determine the value of successor states.

        The depth data member (set in the constructor) determines the maximum depth of the game 
        tree that gets explored before estimating the state utilities using the evaluation() 
        function.  If depth is 0, no traversal is performed, and minimax returns the results of 
        a call to evaluation().  If depth is None, the entire game tree is traversed.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the (possibly estimated) minimax utility value of the state
        """

        return self.minimax_depth(state,depth_limit=None)
        
    

    def evaluation(self, state):
        """Estimate the utility value of the game state based on features.

        Gets called by minimax() once the depth limit has been reached.  
        N.B.: This method must run in "constant" time for all states!

        Args:
            state: a connect383.GameState object representing the current board

        Returns: a heuristic estimate of the utility value of the state
        """
        def streaks(lst):
            """Get the lengths of all the streaks of the same element in a sequence."""
            rets = []  # list of (element, length) tuples
            prev = lst[0]
            curr_len = 1
            for curr in lst[1:]:
                if curr == prev:
                    curr_len += 1
                else:
                    rets.append((prev, curr_len))
                    prev = curr
                    curr_len = 1
            rets.append((prev, curr_len))
            return rets
        p1_score = 0
        p2_score = 0
        
        for go in + state.get_diags() + state.get_rows() + state.get_cols():
            for elt, length in streaks(go):
                if (elt == 1) and (length==2): #max player
                    for i in go:
                        if (i < len(go)-2):
                            if (go[i+1]==1 and go[i+2]==1 and (go[i]==None or go[i]==1)):
                                if i+3 < len(go) and (go[i+3] == None or go[i+3] == 1):
                                    p1_score+=2
                                else:
                                    p1_score+=1 
                            if (go[i]==1 and go[i+1]==1 and (go[i+2] == None or go[i+2] == 1)): 
                                p1_score+=1
                elif (elt == -1) and (length==2): #min player
                    for i in go:
                        if (i < len(go)-2):
                            if (go[i+1]==-1 and go[i+2]==-1 and (go[i]==None or go[i]==-1)):
                                if i+3 < len(go) and (go[i+3] == None or go[i+3] == -1):
                                    p2_score+=2
                                else:
                                    p2_score+=1 
                            if (go[i]==-1 and go[i+1]==-1 and (go[i+2] == None or go[i+2] == -1)): 
                                p2_score+=1
        
        return p1_score - p2_score


class AltMinimaxLookaheadAgent(MinimaxAgent):
    """Alternative heursitic agent used for testing."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit
    
    

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state."""
        #
        # Fill this in, if it pleases you.
        #
        return 19  # Change this line, unless you have something better to do.


class MinimaxPruneAgent(MinimaxAgent):
    """Computer agent that uses minimax with alpha-beta pruning to select the best move.
    
    Hint: Consider what you did for MinimaxAgent.  What do you need to change to prune a
    branch of the state space? 
    """
    
    def minimax(self, state):
        """Determine the minimax utility value the given state using alpha-beta pruning.

        The value should be equal to the one determined by MinimaxAgent.minimax(), but the 
        algorithm should do less work.  You can check this by inspecting the value of the class 
        variable GameState.state_count, which keeps track of how many GameState objects have been 
        created over time.  This agent does not have a depth limit.

        N.B.: When exploring the game tree and expanding nodes, you must consider the child nodes
        in the order that they are returned by GameState.successors().  That is, you cannot prune
        the state reached by moving to column 4 before you've explored the state reached by a move
        to column 1 (we're trading optimality for gradeability here).

        Args: 
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """

        if state.is_full():
            return state.utility()

        return self.alphabeta(state,-math.inf,math.inf)



        
    def alphabeta(self, state,alpha, beta):
        """This is just a helper method for minimax(). Feel free to use it or not."""
        next_p = state.next_player() 
        if next_p == 1:
            infin = math.inf
            holder = state.successors()
            for move in holder:
                temp1 = self.alphabeta(move[1],alpha,beta)
                infin = min(temp1,infin)
                beta = min(infin,beta)
                if beta <= alpha:
                    break
            return infin
        else:
            infin = -math.inf
            holder = state.successors()
            for move in holder:
                temp2 = self.alphabeta(move[1],alpha,beta)
                infin = min(temp2,infin)
                beta = min(infin,beta)
                if beta <= alpha:
                    break
            return infin

def get_agent(tag):
    if tag == 'random':
        return RandomAgent()
    elif tag == 'human':
        return HumanAgent()
    elif tag == 'mini':
        return MinimaxAgent()
    elif tag == 'prune':
        return MinimaxPruneAgent()
    elif tag.startswith('look'):
        depth = int(tag[4:])
        return MinimaxLookaheadAgent(depth)
    elif tag.startswith('alt'):
        depth = int(tag[3:])
        return AltMinimaxLookaheadAgent(depth)
    else:
        raise ValueError("bad agent tag: '{}'".format(tag))       
