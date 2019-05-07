import random, math, copy
from sample_players import DataPlayer


class CustomPlayer_baseline(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.minimax(state, depth=3))

    def minimax(self, state, depth):

        def min_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1))
            return value

        def max_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1))
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
    
### Implementing MONTE CARLO SEARCH AGENT
class CustomAgent(DataPlayer):
    """
    Implementing an agent to play knight's Isolation with Monte Carlo Tree Search
    """

    def mcts(self, state):
        
        root = MCTS_Node(state)
        if root.state.terminal_test():
            return random.choice(state.actions())
        for _ in range(iter_limit):
            child = tree_policy(root)
            if not child:
                continue
            reward = default_policy(child.state)
            backup(child, reward)

        idx = root.children.index(best_child(root))
        return root.children_actions[idx]

    def get_action(self, state):
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.mcts(state))
    
    

class MCTS_Node():
    def __init__(self, state, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.children_actions = []
        self.parent = parent

    def add_child(self, child_state, action):
        child = MCTS_Node(child_state, self)
        self.children.append(child)
        self.children_actions.append(action)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_explored(self):
        return len(self.children_actions) == len(self.state.actions())


FACTOR = 1.0
iter_limit = 100

# def mcts(state):
#     root = MCTS_Node(state)
#     for _ in range(iter_limit):
#         child = tree_policy(root)
#         reward = default_policy(child.state)
#         backup(child,reward)
#
#     idx = root.children.index(best_child(root))
#     return root.children_actions[idx]

def tree_policy(node):
    """
    Select a leaf node.
    If not fully explored, return an unexplored child node.
    Otherwise, return the child node with the best score.
    :param node:
    :return: node
    """
    while not node.state.terminal_test():
        if not node.fully_explored():
            return expand(node)
        node = best_child(node)
    return node


def expand(node):
    tried_actions = node.children_actions
    legal_actions = node.state.actions()
    for action in legal_actions:
        if action not in tried_actions:
            new_state = node.state.result(action)
            node.add_child(new_state, action)
            return node.children[-1]


def best_child(node):
    """
    Find the child node with the best score.
    :param node:
    :return: node;
    """
    best_score = float("-inf")
    best_children = []
    for child in node.children:
        exploit = child.reward / child.visits
        explore = math.sqrt(2.0 * math.log(node.visits) / child.visits)
        score = exploit + FACTOR * explore
        if score == best_score:
            best_children.append(child)
        elif score > best_score:
            best_children = [child]
            best_score = score
    # if len(best_children) == 0:
    #     print("WARNING - RuiZheng, there is no best child")
    #     return None
    return random.choice(best_children)


def default_policy(state):
    """
    Randomly search the descendant of the state, and return the reward
    :param state:
    :return: int
    """
    init_state = copy.deepcopy(state)
    while not state.terminal_test():
        action = random.choice(state.actions())
        state = state.result(action)

    # let the reward be 1 for the winner, -1 for the loser
    # if the init_state.player() wins, it means the action that leads to
    # init_state should be discouraged, so reward = -1.
    return -1 if state._has_liberties(init_state.player()) else 1


def backup(node, reward):
    """
    Backpropagation
    Use the result to update information in the nodes on the path.
    :param node:
    :param reward: int
    :return:
    """
    while node != None:
        node.update(reward)
        node = node.parent
        reward *= -1

### The agent selected is MONTE CARLO SEARCH AGENT

#CustomPlayer = CustomPlayer_baseline # Minimax Agent
CustomPlayer = CustomAgent # Monte Carlo agent