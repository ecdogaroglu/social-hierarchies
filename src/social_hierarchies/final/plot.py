"""Functions plotting results."""

import numpy as np
import networkx as nx
import quantecon as qe

from social_hierarchies.dynamics.graph import (
    compute_rcc_index,
    _create_rcc_graph,
    find_edmonds_arboresence,
    _edges_sp_graph,
)


def plot_stationary_distr(states, trans_matrix,index=0):
    """Create the plot for the stationary distribution.

    Args:
        states (numpy.ndarray): Three dimensional array representing the state space of the stochastic process,
                            where each state is a play history of size m.
                            Dimensions are ((num_act**num_players)**m) x m x num_players
        trans_matrix (numpy.ndarray): The transition matrix of the process.
        index (int): Index of the stationary distribution. Default is set to the first one, other distributions can be accessed when there are multiple ones.

    Returns:
        numpy.ndarray : Linear space for plotting.
        numpy.ndarray : Probability vector of the stationary distribution.

    """
    num_states = states.shape[0]

    mc = qe.MarkovChain(trans_matrix)
    s_distr = mc.stationary_distributions[index]
    x = np.linspace(0, num_states - 1, num_states)

    return x, s_distr


def plot_rcc_graph(G, states, k, num_act, payoffs, trans_matrix_un):
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
    G_rcc = _create_rcc_graph(states, k, num_act, rcc_index, G, payoffs)

    indices = list(G_rcc.nodes())
    labels = {}
    for i in indices:
        labels[i] = states[i] + np.ones((states.shape[1], states.shape[2]))

    edge_labels = nx.get_edge_attributes(G_rcc, "weight")

    return G_rcc, labels, edge_labels


def plot_shortest_path_example(G, states, k, num_act, payoffs):
    """Create the plot for an example shortest path between the recurrent communication classes.

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

    G_sp_ex = nx.DiGraph()
    G_sp_ex.add_weighted_edges_from(_edges_sp_graph(0,255,states,G, k, num_act, payoffs))

    indices = list(G_sp_ex.nodes())
    sp_labels = {}
    for i in indices:
        sp_labels[i] = states[i] + np.ones((states.shape[1], states.shape[2]))
    sp_edge_labels = nx.get_edge_attributes(G_sp_ex, "weight")
   
    return G_sp_ex, sp_labels, sp_edge_labels


def plot_edmonds_arboresence(G, states, k, num_act, payoffs, trans_matrix_un):
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
    arb = find_edmonds_arboresence(G, states, trans_matrix_un, k, num_act, payoffs)
    indices = list(arb.nodes())
    labels = {}
    for i in indices:
        labels[i] = states[i] + np.ones((states.shape[1], states.shape[2]))
    edge_labels = nx.get_edge_attributes(arb, "weight")

    return arb, labels, edge_labels
