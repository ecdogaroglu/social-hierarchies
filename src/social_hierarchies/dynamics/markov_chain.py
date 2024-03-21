"""Functions creating markov chains and extracting their properties."""

import numpy as np
import quantecon as qe
import pandas as pd

from social_hierarchies.dynamics.probabilities import pl_pos_is_s, pl_pos_is_s_by_mistake


def compute_trans_matrix_unperturbed(states, k, num_act, payoffs):
    """Compute the transition matrix for the given state according to the unperturbed
    process without mistakes.

    Args:
        states (numpy.ndarray): Three dimensional array representing the state space of the stochastic process,
                            where each state is a play history of size m.
                            Dimensions are ((num_act**num_players)**m) x m x num_players
        k (int): The sample size according to which play histories are sampled.
        num_act (int): Number of actions in the game. Different number of actions for each player is not supported.
        payoffs (numpy.ndarray): The payoff matrix of the game with dimensions num_act x num_act x num_players. Only num_players = 2 is supported.

    Returns:
        numpy.ndarray: The transition matrix for the given state space and unperturbed process with dimensions num_states x num_states.
                                        Each entry is a float in the interval [0,1], where each row must sum to 1. (test)

    """
    num_states = states.shape[0]
    transition_matrix_up = np.zeros((num_states, num_states))

    for i in range(num_states):
        for j in range(num_states):
            transition_matrix_up[i, j] = _trans_prob_unperturbed(
                states[i], states[j], k, num_act, payoffs
            )

    return transition_matrix_up


def compute_trans_matrix_perturbed(states, k, epsilon, num_act, payoffs):
    """Compute the transition matrix for the given state according to the perturbed
    process with mistakes.

    Args:
        states (numpy.ndarray): Three dimensional array representing the state space of the stochastic process,
                            where each state is a play history of size m.
                            Dimensions are ((num_act**num_players)**m) x m x num_players
        k (int): The sample size according to which play histories are sampled.
        epsilon (float): The probability by which a player experiments and takes a random action. Random actions are picked according to the uniform distribution.
                            The paper shows that this probability distribution is irrelavant for the results as long as it has full support.
                            Must be in the interval [0,1].
        num_act (int): Number of actions in the game. Different number of actions for each player is not supported.
        payoffs (numpy.ndarray): The payoff matrix of the game with dimensions num_act x num_act x num_players. Only num_players = 2 is supported.

    Returns:
        numpy.ndarray: The transition matrix for the given state space and unperturbed process with dimensions num_states x num_states.
                                        Each entry is a float in the interval [0,1], where each row must sum to 1. (test)

    """
    num_states = states.shape[0]
    transition_matrix_p = np.zeros((num_states, num_states))

    for i in range(num_states):
        for j in range(num_states):
            transition_matrix_p[i, j] = _trans_prob_perturbed(
                states[i], states[j], k, epsilon, num_act, payoffs
            )
    return transition_matrix_p


def _trans_prob_unperturbed(pre, suc, k, num_act, payoffs):
    """Compute the transition probability from one state to another according to the
    unpertubed process with no mistakes.

    Args:
        pre (numpy.ndarray): The current state of the process. Two dimensional array, typically an element of the state space.
        suc (numpy.ndarray): The next state for which the transition probability is calculated. Two dimensional array, typically an element of the state space.
        k (int): The sample size according to which play histories are sampled.
        num_act (int): Number of actions in the game. Different number of actions for each player is not supported.
        payoffs (numpy.ndarray): The payoff matrix of the game with dimensions num_act x num_act x num_players. Only num_players = 2 is supported.

    Returns:
        float: The probability by which the state will transition from the current state to the next state.
                                    Can only be positive if next state is a successor of current state.
                                    Must be in the interval [0,1].

    """
    out = 0
    is_suc = []
    
    for i in range(pre.shape[0]):
        is_suc.append((pre[i][1:] == suc[i][:-1]).all())

        if sum(is_suc) == pre.shape[0]:
                level_out = []

                for j in range(pre.shape[0]):
                    level_prob = pl_pos_is_s(0,j,suc[j][-1][0],payoffs,pre) * pl_pos_is_s(1,j,suc[j][-1][1],payoffs,pre)
                    level_out.append(level_prob)
                    
                level_out = np.array(level_out)
                out = np.prod(level_out)

    return out


def _trans_prob_perturbed(pre, suc, k, epsilon, num_act, payoffs):  # doesnt support n>2
    """Compute the transition probability from one state to another according to the
    perturbed process with mistakes.

    Methodology:
    There are four possible cases that the next states can be achieved given that it's a successor of the current state.
    1) Both players optimize and their corresponding actions in the "next" state are in fact among the best responses.
    2) First player optimizes and her action in the "next" state is a best response, while second player experiments and her corresponding action is taken by chance.
    3) Second player optimizes and her action in the "next" state is a best response, while first player experiments and her corresponding action is taken by chance.
    4) Both players experiment and their corresponding actions are taken by chance.

    The sum of these 4 disjoint events correspond to the overall probability that the "next" state can be achieved.

    Args:
        pre (numpy.ndarray): The current state of the process. Two dimensional array, typically an element of the state space.
        suc (numpy.ndarray): The next state for which the transition probability is calculated. Two dimensional array, typically an element of the state space.
        k (int): The sample size according to which play histories are sampled.
        epsilon (float): The probability by which a player experiments and takes a random action. Random actions are picked according to the uniform distribution.
                            The paper shows that this probability distribution is irrelavant for the results as long as it has full support.
                            Must be in the interval [0,1].
        num_act (int): Number of actions in the game. Different number of actions for each player is not supported.
        payoffs (numpy.ndarray): The payoff matrix of the game with dimensions num_act x num_act x num_players. Only num_players = 2 is supported.


    Returns:
        float: The probability by which the state will transition from the current state to the next state.
                Can only be positive if next state is a successor of current state.
                Must be in the interval [0,1].

    """
    out = 0
    is_suc = []
    
    for i in range(pre.shape[0]):
        is_suc.append((pre[i][1:] == suc[i][:-1]).all())

        if sum(is_suc) == pre.shape[0]:
                no_mistake = []
                mistake = []

                for a in range(pre.shape[0]):
                    add = pl_pos_is_s(0,a,suc[a][-1][0],payoffs,pre) * pl_pos_is_s(1,a,suc[a][-1][1],payoffs,pre)
                    no_mistake.append(add)

                for a in range(pre.shape[0]):
                    add = pl_pos_is_s_by_mistake(0,a,suc[a][-1][0],payoffs,pre) * pl_pos_is_s_by_mistake(1,a,suc[a][-1][1],payoffs,pre)
                    mistake.append(add)
                
                no_mistake_total = np.prod(np.array(no_mistake))
                mistake_total = np.prod(np.array(mistake))

                out =  (1-epsilon) * no_mistake_total + epsilon * mistake_total

    return out


def compute_stoch_stab_states(transition_matrix):
    """Compute the stochastically stable states with the convention that the states has
    probability more that 1% on the stationary distribution.

    Args:
        states (numpy.ndarray): Three dimensional array representing the state space of the stochastic process,
                            where each state is a play history of size m.
                            Dimensions are ((num_act**num_players)**m) x m x num_players
        transition_matrix (numpy.ndarray): The transition matrix for which the recurrent communication classes are be computed.

    Returns:
        pandas.DataFrame: Data frame containing the stochastically stable states and their probability in two columns.

    """
    mc = qe.MarkovChain(transition_matrix)
    s_dist = mc.stationary_distributions[0]
    ss_states_df = _s_dist_to_ss_states(s_dist)

    return ss_states_df


def _s_dist_to_ss_states(s_dist):
    """Compute the stochastically stable states from a stationary distribution. States
    with probability greater than 0.01 are taken to be stable.

    Args:
        s_dist (numpy.ndarray): The statationary distribution for which the stochastically stable states are returned.

    Returns:
        pandas.DataFrame: Data frame containing the stochastically stable states and their probability in two columns.

    """
    ind = np.argwhere(s_dist > 0.01)

    indices = []
    for i in range(ind.shape[0]):
        indices.append(int(ind[i][0]))

    ss_states = {}
    for i in range(len(indices)):
        ss_states[i] = [indices[i], s_dist[indices[i]]]

    ss_states_df = pd.DataFrame.from_dict(
        ss_states, orient="index", columns=["state index", "probability"]
    )

    return ss_states_df


def compute_rcc_index(transition_matrix):
    """Compute the recurrent communication classes of a transition matrix.

    Args:
        transition_matrix (numpy.ndarray): The transition matrix for which the recurrent communication classes should be computed.

    Returns:
        list: The indices of the recurrent communication classes according to the ordering of the state space.

    """

    mc = qe.MarkovChain(transition_matrix)

    rcc_index = []
    for i in range(mc.num_recurrent_classes):
        rcc_index.append(mc.recurrent_classes_indices[i][0])

    return rcc_index


def ___index_to_state(states, indices):
    """Transform state indices to states.

    Args:
        states (numpy.ndarray): Three dimensional array representing the state space of the stochastic process,
                            where each state is a play history of size m.
                            Dimensions are ((num_act**num_players)**m) x m x num_players
        indices (list or int): Indicies of the states to be transformed.

    Returns:
        numpy.ndarray : Array containing the corresponding states.

    """
    if type(indices) == int:
        indcs = []
        indcs.append(indices)
    if type(indices) == list:
        indcs = indices

    state_list = []
    for i in range(len(indcs)):
        id = indcs[i]
        state_list.append(states[id])

    state_list = np.array(state_list)

    return state_list
