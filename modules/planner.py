import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any



class Planner:
    """
    A simple planner.
    """

    def __init__(
            self,
            beta_move: float = 4.0,
            eps_move: float = 0.02,
            learning_rate: float = 0.1,
        ):
        """
        Args:
            beta_move, eps_move
                inverse temperature and lapse for sampling moves from action values.
            learning_rate
                learning rate for Q-value updates during backpropagation.
        """

        # initialize parameters
        self.beta_move = beta_move
        self.eps_move = eps_move
        self.learning_rate = learning_rate

        # internal per-problem state (initialized at simulate(...) start)
        self.root = 0
        self.children: Dict[int, List[int]] = {}
        self.rewards: Dict[int, float] = {}
        self.is_terminal = lambda s: len(self.children.get(s, [])) == 0

        # tree bookkeeping (partial view)
        self.expanded: set[int] = set() # nodes whose children are currently revealed
        self.parents: Dict[int, Optional[int]] = {}
        self.frontier: set[int] = set() # nodes known but not expanded yet

        # value caches
        self.q: Dict[int, float] = {}
        self.g: Dict[int, float] = {} # path-sum reward from root to node

        # visit counts
        self.n_visits: Dict[int, int] = {}
        self.n_fix: int = 0

        # trace (optional)
        self.trace: List[Dict[str, Any]] = []


    # ---------- q value backpropagation ----------
    def update_q(
            self,
            node: int,
        ):
        """
        Backpropagate value along a path using MCTS-style updates.
        """

        # make sure expanded
        if node not in self.expanded:
            return

        # initialize Q value if not present
        if node not in self.q:
            self.q[node] = 0.0

        # eligible children = expanded children only (strong info gating)
        children = self.children.get(node, []) # [c for c in self.children.get(node, []) if c in self.expanded]

        # get reward
        r = self.rewards.get(node, 0.0)

        if not children:
            target = r
        else:
            best_child_q = max(self.q.get(c, 0.0) for c in children)
            target = r + best_child_q

        self.q[node] += self.learning_rate * (target - self.q[node])



    # ---------- g value update ----------
    def update_g(
            self,
            node: int
        ):
        """
        Update g value (path-sum reward from root) for a specific node.
            If the node has a parent, g(node) = g(parent) + reward(parent).
        """
        
        # root node always has g = 0
        if node == self.root:
            self.g[node] = 0.0
            return
        
        # if node has a parent, update based on parent's g value
        parent = self.parents.get(node)
        if parent is not None:
            self.g[node] = self.g.get(parent, 0.0) + self.rewards.get(parent, 0.0)
    

    # ---------- initialization ----------
    def init_problem(
            self,
            children: Dict[int, List[int]],
            rewards: Dict[int, float],
            root: int = 0
        ):
        """
        Initialize problem.
        """

        self.children = children
        self.rewards = rewards
        self.root = root

        # reset per-problem state
        self.expanded = {root} # nodes whose children are revealed
        self.parents = {root: None} # map nodes to their parents (if nothing, the node hasn't been seen)
        self.frontier = set(children.get(root, [])) # nodes known but not expanded (candidates for next look)

        # values & visits
        self.q = {}
        self.g = {root: 0.0}
        self.n_visits = {}
        self.n_fix = 0
        self.trace = []

        # initialize frontier parent links
        for child in self.frontier:
            self.parents[child] = root

            # update g value
            self.update_g(child)
        
        # initialize Q value for root to 0
        self.q[root] = 0.0


    # ---------- fixation ----------
    def expand(
            self,
            node: int
        ):
        """
        Reveal children of node (if any) into the partial tree and do global backup.
        """

        # return if already expanded
        if node in self.expanded:
            return
        
        # add to the expanded set
        self.expanded.add(node)

        children = self.children.get(node, [])
        for child in children:
            # if not added to the partial tree
            if child not in self.parents:
                # link new child to parent
                self.parents[child] = node

                # add to frontier
                self.frontier.add(child)

                # initialize g value
                self.update_g(child)

    
    def look(
            self,
            node: int
        ):
        """
        Perform a fixation (look) at node:
          1. count visit
          2. update g value for the node
          3. expand if needed
          4. backpropagate vallue from this node to root
        """

        # update visit count
        self.n_fix += 1
        self.n_visits[node] = self.n_visits.get(node, 0) + 1

        # update g value every time we look at a node
        self.update_g(node)

        # if in the partial tree but not expanded
        if node not in self.expanded and node in self.parents:
            self.expand(node)

        # if already expanded but had hidden children in full problem, reveal them
        else:
            children = self.children.get(node, [])
            if any(child not in self.parents for child in children):
                self.expand(node)

        # update q value
        self.update_q(node)

        # record trace
        self.trace.append({'event': 'LOOK', 'node': node, 'n_fix': self.n_fix})
        

    # ---------- move ----------
    def sample_move(
            self,
            node: int
        ) -> Optional[int]:
        """
        Choose a child to move to from node, proportional to Q(child).
        """

        # get children
        children = self.children.get(node, [])

        # return None if no child
        if len(children) == 0:
            return None
        
        # compute q values of the children
        q_children = np.array([self.q.get(child, 0.0) for child in children], dtype = float)

        # randomly choose a child to move
        return choice(children, q_children, self.beta_move, self.eps_move)
    

    def move(self) -> Optional[int]:
        """
        Move until reaching a terminal.
        """

        node = self.root
        chosen_path = []
        cum_reward = 0

        while not self.is_terminal(node):
            child = self.sample_move(node)
            if child is not None:
                cum_reward += self.rewards[child]
                node = child
                chosen_path.append(node)
        
        return cum_reward, chosen_path






def softmax(
        x: np.ndarray,
        beta: float,
        eps: float
    ) -> np.ndarray:
    """
    Softmax with inverse temperature beta and lapse eps (uniform mixing).
    """

    if x.size == 0:
        return x
    
    # subtract max for numerical stability
    z = beta * (x - np.max(x))
    p = np.exp(z)
    p = p / p.sum()

    # ensure numerical stability
    if eps > 0.0:
        p = (1 - eps) * p + eps * (1.0 / len(p))

    return p


def choice(
        items: List[Any],
        scores: np.ndarray,
        beta: float,
        eps: float
    ) -> Any:
    """
    Sample one item proportional to softmax(scores * beta) with lapse eps.
    """

    probs = softmax(scores, beta, eps)
    idx = np.random.choice(len(items), p = probs)

    return items[idx] 