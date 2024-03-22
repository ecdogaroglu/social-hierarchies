"""Functions creating and analyzing directed graphs."""

import networkx as nx
import math
import numpy as np

from social_hierarchies.dynamics.probabilities import pl_pos_is_s
from social_hierarchies.dynamics.markov_chain import compute_rcc_index, ___index_to_state


def find_edmonds_arboresence(G, states, trans_matrix_un, k, num_act, payoffs):
    """Create the directed graph for recurrent communication classes of the unperturbed
    process and find max arboresence (RCC with minimum stochastic potential) using the
    Edmond (1967)'s algorithm. Returns only one such arboresence even if there are
    multiple. To find all such arboresence, see the function find_min_stoch_pot.

    Args:
        states (numpy.ndarray): Three dimensional array representing the state space of the stochastic process,
                            where each state is a play history of size m.
                            Dimensions are ((num_act**num_players)**m) x m x num_players
        trans_matrix_un (numpy.ndarray): The transition matrix of the unperturbed process.
        k (int): The sample size according to which play histories are sampled.
        num_act (int): Number of actions in the game. Different number of actions for each player is not supported.
        payoffs (numpy.ndarray): The payoff matrix of the game with dimensions num_act x num_act x num_players. Only num_players = 2 is supported.

    Returns:
        networkx.classes.digraph.DiGraph : The arboresence whose root is corresonding to the state with least stochastic potential.

    """
    rcc_index = compute_rcc_index(trans_matrix_un)
    G_rcc = _create_rcc_graph(states, k, num_act, rcc_index, G, payoffs)

    ed = nx.algorithms.tree.branchings.Edmonds(G_rcc)
    arb = ed.find_optimum(attr="weight", style="arborescence", kind="min")

    return arb


def _create_directed_graph(states, k, num_act, payoffs):
    """Create the directed graph for the state space, where weights correspond to
    "resistances".

    Args:
        states (numpy.ndarray): Three dimensional array representing the state space of the stochastic process,
                            where each state is a play history of size m.
                            Dimensions are ((num_act**num_players)**m) x m x num_players
        k (int): The sample size according to which play histories are sampled.
        num_act (int): Number of actions in the game. Different number of actions for each player is not supported.
        payoffs (numpy.ndarray): The payoff matrix of the game with dimensions num_act x num_act x num_players. Only num_players = 2 is supported.

    Returns:
        networkx.classes.digraph.DiGraph : The directed graph of the unpurturbed process.

    """

    num_states = states.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(num_states))

    edges = []
    for i in range(num_states):
        for j in range(num_states):
            edges.append(
                (i, j, __resistance(states[i], states[j], k, num_act, payoffs))
            )

    G.add_weighted_edges_from(edges)

    return G


def __resistance(source, target, k, num_act, payoffs):
    """Calculate the "resistance" of transition from one state to another according to
    the unperturbed process.

    Args:
        source (numpy.ndarray): The current state of the process. Two dimensional array, typically an element of the state space.
        target (numpy.ndarray): The target state for which the resistance of transition is calculated. Two dimensional array, typically an element of the state space.
        k (int): The sample size according to which play histories are sampled.
        num_act (int): Number of actions in the game. Different number of actions for each player is not supported.
        payoffs (numpy.ndarray): The payoff matrix of the game with dimensions num_act x num_act x num_players. Only num_players = 2 is supported.

    Returns:
        int or float : Sum of mistakes (int); or infinity (float) if target is not a successor of the source.

    """
    is_suc = []

    cond=False
    for i in range(source.shape[0]):
        is_suc.append((source[i][1:] == target[i][:-1]).all())
        cond = sum(is_suc) == source.shape[0]
    
    if cond:
        mistakes = []
        for i in range(source.shape[0]):
            s = target[i][-1]
            for j in range(s.shape[0]):        
                mistakes.append(pl_pos_is_s(j,i,s[j],payoffs,source, k, num_act) == 0)
            res = sum(mistakes)

    if not cond:
        res = math.inf

    return res




def _create_rcc_graph(states, k, num_act, rcc_index, G, payoffs):
    """Create the directed graph for the recurrent communication classes, where weights
    correspond to "total resistances".

    Args:
        states (numpy.ndarray): Three dimensional array representing the state space of the stochastic process,
                            where each state is a play history of size m.
                            Dimensions are ((num_act**num_players)**m) x m x num_players
        k (int): The sample size according to which play histories are sampled.
        num_act (int): Number of actions in the game. Different number of actions for each player is not supported.
        rcc_index (list): Indicies of the recurrent communication classes according to the ordering of the state space.
        G (networkx.classes.digraph.DiGraph) : Directed graph for the whole state space.
        payoffs (numpy.ndarray): The payoff matrix of the game with dimensions num_act x num_act x num_players. Only num_players = 2 is supported.

    Returns:
        networkx.classes.digraph.DiGraph : The directed graph of the recurrent communication classes.

    """
    G_rcc = nx.DiGraph()
    G_rcc.add_nodes_from(rcc_index)

    edges_rcc = []
    for i in rcc_index:
        for j in rcc_index:
            edges_rcc.append(
                (i, j, __total_resistance(i, j, states, k, num_act, G, payoffs))
            )

    G_rcc.add_weighted_edges_from(edges_rcc)

    return G_rcc


def __total_resistance(source, target, states, k, num_act, G, payoffs):
    """Calculate the "total resistance" of transition from one state to another
    according to the unperturbed process.

    Methodology:
    Find the shortest path form one RCC to another in the original graph.
    Sum the resistances along the path.

    Args:
        source (numpy.ndarray): The current state of the process. Two dimensional array, typically an element of the state space.
        target (numpy.ndarray): The target state for which the total resistance of transition is calculated. Two dimensional array, typically an element of the state space.
        states (numpy.ndarray): Three dimensional array representing the state space of the stochastic process,
                                where each state is a play history of size m.
                                Dimensions are ((num_act**num_players)**m) x m x num_players
        k (int): The sample size according to which play histories are sampled.
        num_act (int): Number of actions in the game. Different number of actions for each player is not supported.
        G (networkx.classes.digraph.DiGraph) : Directed graph for the whole state space.
        payoffs (numpy.ndarray): The payoff matrix of the game with dimensions num_act x num_act x num_players. Only num_players = 2 is supported.

    Returns:
        int : Sum of resistances along the shortest path from one RCC to another.

    """
    s_p = nx.algorithms.shortest_path(G, source= f"{source}", target=f"{target}", weight="weight")
    res = 0
    for i in range(len(s_p) - 1):
        pre = int(s_p[i])
        suc = int(s_p[i + 1])
        res += __resistance(states[pre], states[suc], k, num_act, payoffs)

    res = int(res)

    return res


def find_states_with_min_stoch_pot(G, states, trans_matrix_un, k, num_act, payoffs):
    """Create the directed graph for recurrent communication classes of the unperturbed
    process and find all RCCs with minimum stochastic potential using the Edmond
    (1967)'s algorithm.

    Args:
        states (numpy.ndarray): Three dimensional array representing the state space of the stochastic process,
                            where each state is a play history of size m.
                            Dimensions are ((num_act**num_players)**m) x m x num_players
        trans_matrix_un (numpy.ndarray): The transition matrix of the unperturbed process.
        k (int): The sample size according to which play histories are sampled.
        num_act (int): Number of actions in the game. Different number of actions for each player is not supported.
        payoffs (numpy.ndarray): The payoff matrix of the game with dimensions num_act x num_act x num_players. Only num_players = 2 is supported.

    Returns:
        networkx.classes.digraph.DiGraph : The arboresence whose root is corresonding to the state with least stochastic potential.

    """
    rcc_index = compute_rcc_index(trans_matrix_un)
    G_rcc = _create_rcc_graph(states, k, num_act, rcc_index, G, payoffs)

    arbs, root_states = _root_states(states, G_rcc, rcc_index)

    return arbs, root_states


def _root_states(states, G_rcc, rcc_index):
    """Find the arboresences with minimum weights and their root states.

    Args:
        states (numpy.ndarray): Three dimensional array representing the state space of the stochastic process,
                            where each state is a play history of size m.
                            Dimensions are ((num_act**num_players)**m) x m x num_players
        G_rcc (networkx.classes.digraph.DiGraph) : Directed graph for RCCs.
        rcc_index (list): Indicies of the recurrent communication classes according to the ordering of the state space.

    Returns:
        list : List containing arboresences (networkx.classes.digraph.DiGraph objects) with minimum weight.
        numpy.ndarray : Array containing the roots of these arboresences i.e. the states with minimum stochastic potential.

    """
    arbs_weight = __arbs_total_weight(G_rcc)
    itr = iter(nx.algorithms.tree.ArborescenceIterator(G_rcc))
    min_arb_index = [
        kv[0] for kv in arbs_weight.items() if kv[1] == min(arbs_weight.values())
    ]

    root_states = []
    for i in range(len(min_arb_index)):
        root_states.append(__find_root_state(states, next(itr), rcc_index))

    root_states = np.array(root_states)

    itr2 = iter(nx.algorithms.tree.ArborescenceIterator(G_rcc))
    arbs = []
    for i in range(len(min_arb_index)):
        arbs.append(next(itr2))

    return arbs, root_states


def __arbs_total_weight(G_rcc):
    """Find total weights of all possible arboresences and store them in a dictionary.

    Args:
        G_rcc (networkx.classes.digraph.DiGraph) : Directed graph for RCCs.

    Returns:
        dict : Dictionary containing the indices of arboresences and their total weights.

    """
    itr = iter(nx.algorithms.tree.ArborescenceIterator(G_rcc))

    dict = {}
    for index, item in enumerate(itr):
        dict[index] = nx.algorithms.tree.branching_weight(
            item, attr="weight"
        )  # which index has how much weight

    return dict


def __find_root_state(states, arb, rcc_index):
    """Find the root state of a given arboresence.

    Args:
        states (numpy.ndarray): Three dimensional array representing the state space of the stochastic process,
                            where each state is a play history of size m.
                            Dimensions are ((num_act**num_players)**m) x m x num_players
        arb (networkx.classes.digraph.DiGraph) : The arboresence whose root is found.
        rcc_index (list): Indicies of the recurrent communication classes according to the ordering of the state space.

    Returns:
        numpy.ndarray : Array containing the root state of these arboresence.

    """
    edges = np.array(list((arb.edges)))

    targets = []
    for i in range(edges.shape[0]):
        targets.append(int(edges[i][1]))

    targets = np.array(targets)
    root_index = list(np.setdiff1d(rcc_index, targets))
    root_states = ___index_to_state(states, root_index)

    return root_states


def _edges_sp_graph(source,target, states, G, k, num_act, payoffs):
    sp = nx.algorithms.shortest_path(G, source= f"{source}", target=f"{target}", weight="weight")
    res = 0
    plot = []
    for i in range(len(sp)-1):
        pre = int(sp[i])
        suc = int(sp[i + 1])
        res = __resistance(states[pre],states[suc], k, num_act, payoffs)
        plot.append((pre,suc,res))
    
    return plot