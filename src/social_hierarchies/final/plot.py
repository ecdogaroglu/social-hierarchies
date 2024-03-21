"""Functions plotting results."""

import numpy as np
import networkx as nx
import quantecon as qe

from social_hierarchies.dynamics.graph import (
    compute_rcc_index,
    _create_directed_graph,
    _create_rcc_graph,
    find_edmonds_arboresence,
)


def plot_stationary_distr(states, trans_matrix_p):
    """Create the plot for the stationary distribution.

    Args:
        states (numpy.ndarray): Three dimensional array representing the state space of the stochastic process,
                            where each state is a play history of size m.
                            Dimensions are ((num_act**num_players)**m) x m x num_players
        trans_matrix_up (numpy.ndarray): The transition matrix of the perturbed process.

    Returns:
        numpy.ndarray : Linear space for plotting.
        numpy.ndarray : Probability vector of the stationary distribution.

    """
    num_states = states.shape[0]

    mc = qe.MarkovChain(trans_matrix_p)
    s_distr = mc.stationary_distributions[0]
    x = np.linspace(0, num_states - 1, num_states)

    return x, s_distr


def plot_rcc_graph(states, k, num_act, payoffs, trans_matrix_un):
    """Create the plot for the RCC graph.

    Args:
        states (numpy.ndarray): Three dimensional array representing the state space of the stochastic process,
                            where each state is a play history of size m.
                            Dimensions are ((num_act**num_players)**m) x m x num_players
        k (int): The sample size according to which play histories are sampled.
        num_act (int): Number of actions in the game. Different number of actions for each player is not supported.
        trans_matrix_un (numpy.ndarray): The transition matrix of the unperturbed process.

    Returns:
        networkx.classes.digraph.DiGraph : The directed graph of the unpurturbed process.
        list : List of numpy arrays contatining the node labels.
        dict : Numpy array containing the edge labels.

    """
    rcc_index = compute_rcc_index(trans_matrix_un)
    G = _create_directed_graph(states, k, num_act, payoffs)
    G_rcc = _create_rcc_graph(states, k, num_act, rcc_index, G, payoffs)

    indices = list(G_rcc.nodes())
    labels = {}
    for i in indices:
        labels[i] = states[i] + np.ones((states.shape[1], states.shape[2]))

    edge_labels = nx.get_edge_attributes(G_rcc, "weight")

    return G_rcc, labels, edge_labels


def plot_edmonds_arboresence(states, k, num_act, payoffs, trans_matrix_un):
    """Create the plot for the Edmond's arboresence.

    Args:
        states (numpy.ndarray): Three dimensional array representing the state space of the stochastic process,
                            where each state is a play history of size m.
                            Dimensions are ((num_act**num_players)**m) x m x num_players
        k (int): The sample size according to which play histories are sampled.
        num_act (int): Number of actions in the game. Different number of actions for each player is not supported.
        payoffs (numpy.ndarray): The payoff matrix of the game with dimensions num_act x num_act x num_players. Only num_players = 2 is supported.

        trans_matrix_un (numpy.ndarray): The transition matrix of the unperturbed process.

    Returns:
        networkx.classes.digraph.DiGraph : The Edmond's arboresence.
        list : List of numpy arrays contatining the node labels.
        dict : Numpy array containing the edge labels.

    """
    arb = find_edmonds_arboresence(states, trans_matrix_un, k, num_act, payoffs)
    indices = list(arb.nodes())
    labels = {}
    for i in indices:
        labels[i] = states[i] + np.ones((states.shape[1], states.shape[2]))
    edge_labels = nx.get_edge_attributes(arb, "weight")

    return arb, labels, edge_labels
