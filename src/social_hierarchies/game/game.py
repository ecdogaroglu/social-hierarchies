"""Functions for defining the state space and calculating the best response
probabilities."""

import numpy as np
import quantecon.game_theory as gt
from itertools import product, permutations


def define_state_space(num_act, num_players, m, l):
    """Define the state space for the stochastic process.

    Args:
        num_act (int): Number of actions in the game. Different number of actions for each player is not supported.
        num_players (int): Number of players in the game. Currently only 2 players are supported.
        m (int): The memory size of players' from which play histories will be sampled.

    Returns:
        numpy.ndarray: Three dimensional array representing the state space of the stochastic process, where each state is a play history of size m.
                    Dimensions are ((num_act**num_players)**m) x num_act x num_players

    """

    actions = np.array(list(range(num_act)))
    possible_plays = np.array(list(product(actions, repeat=num_players)))
    possible_histories = np.array(list(product(possible_plays, repeat=m)))
    states = np.array(list(product(possible_histories, repeat=l)))

    return states


def _best_response_prob(state, strategy, player, k, num_act, payoffs):
    """Compute the probability of a given pure strategy being a best response to a given
    history of play (state).

    Methodology:
    As players only observe a random portion of opponent's play history, the overall probability will be the average over all possible samples.
    First all possible opponent play samples of size k are computed from the given play history of size m.
    Then for ease of computation, possible samplings are reduced to their types and counts of occurences.
    For each possible opponent sample type, best responses are computed and whether the given strategy is among them is stored in a list.
    (Tie breaking is set to false to access the set of all best responses to a given sample.)
    Overall best response probability is then calculated by weighted averaging over the occurence of each opponent sample type.

    Args:
        state (numpy.ndarray): Two dimensional array representing an element of state space.
        strategy (int): The pure strategy (action) index for which the best response probability will be computed.
        k (int): The sample size according to which play histories are sampled.
        num_act (int): Number of actions in the game. Different number of actions for each player is not supported.
        payoffs (numpy.ndarray): The payoff matrix of the game with dimensions num_act x num_act x num_players. Only num_players = 2 is supported.

    Returns:
        float: The overall best response probability of the given pure strategy to the given state (history) of play.
                Must be in the interval [0,1].

    """
    
    game = gt.NormalFormGame(payoffs)
    if player == 0:
        op = 1
    elif player == 1:
        op = 0

    possible_samples = list(permutations(state[:, op], r=k))
    act_dist_for_each_sample = _bincount2d(possible_samples, bins=num_act) / k
    best_responses = []
    for i in range(act_dist_for_each_sample.shape[0]):
        br = np.array(
            game.players[player].best_response(
                act_dist_for_each_sample[i], tie_breaking=False
            )
        )
        if strategy in br:
            best_responses.append(1 / br.shape[0])

    prob_br = sum(best_responses) / act_dist_for_each_sample.shape[0]

    return prob_br


def _bincount2d(arr, bins=None):
    """Count number of occurrences of each value in a 2d array.

    Args:
        arr (numpy.ndarray) : The 2 dimensional array for which the counting will be implemented.

    Returns:
        numpy.ndarray: The 2 dimensional array including the count of occurrences for each row of the original array.

    Credit : https://stackoverflow.com/questions/19201972/can-numpy-bincount-work-with-2d-arrays

    """
    if bins is None:
        bins = np.max(arr) + 1
    count = np.zeros(shape=[len(arr), bins], dtype=np.int64)
    indexing = (np.ones_like(arr).T * np.arange(len(arr))).T
    np.add.at(count, (indexing, arr), 1)

    return count
