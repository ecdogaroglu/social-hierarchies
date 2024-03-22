"""Tasks running the core analyses regarding the dynamics."""

import numpy as np
import pickle
import networkx as nx

from social_hierarchies.config import BLD
from social_hierarchies.dynamics.markov_chain import (
    compute_rcc_index,
    compute_trans_matrix_unperturbed,
    compute_trans_matrix_perturbed,
    compute_stoch_stab_states,
)
from social_hierarchies.dynamics.graph import (
    find_edmonds_arboresence,
    __find_root_state,
    find_states_with_min_stoch_pot,
    _create_directed_graph,
)


def task_save_trans_matrix_unperturbed(
    states_dir=BLD / "python" / "game" / "states.npy",
    parameters_dir=BLD / "python" / "game" / "parameters.pkl",
    produces=BLD / "python" / "dynamics" / "trans_matrix_un.npy",
):
    """Compute the transition matrix of the unperturbed process."""

    with open(parameters_dir, "rb") as fp:
        parameters = pickle.load(fp)

    states = np.load(states_dir)
    k = parameters["k"]
    num_act = parameters["num_act"]
    payoffs = parameters["payoffs"]

    transition_matrix_up = compute_trans_matrix_unperturbed(states, k, num_act, payoffs)

    np.save(produces, transition_matrix_up)


def task_save_trans_matrix_perturbed(
    states_dir=BLD / "python" / "game" / "states.npy",
    parameters_dir=BLD / "python" / "game" / "parameters.pkl",
    produces=BLD / "python" / "dynamics" / "trans_matrix_p.npy",
):
    """Compute the transition matrix of the perturbed process."""

    with open(parameters_dir, "rb") as fp:
        parameters = pickle.load(fp)

    states = np.load(states_dir)
    k = parameters["k"]
    payoffs = parameters["payoffs"]
    epsilon = parameters["epsilon"]
    num_act = parameters["num_act"]

    transition_matrix_p = compute_trans_matrix_perturbed(
        states, k, epsilon, num_act, payoffs
    )

    np.save(produces, transition_matrix_p)

    transition_matrix_p.tofile(
        BLD / "python" / "dynamics" / "trans_matrix_p.csv", sep=","
    )

def task_save_G(
    states_dir=BLD / "python" / "game" / "states.npy",
    parameters_dir=BLD / "python" / "game" / "parameters.pkl",
    produces=BLD / "python" / "dynamics" / "G.net",
):
    """Compute the transition matrix of the perturbed process."""

    with open(parameters_dir, "rb") as fp:
        parameters = pickle.load(fp)

    states = np.load(states_dir)
    k = parameters["k"]
    payoffs = parameters["payoffs"]
    num_act = parameters["num_act"]

    G = _create_directed_graph(states, k, num_act, payoffs)
    nx.write_pajek(G, produces)


def task_save_stoch_stable_states(
    trans_matrix_p_dir=BLD / "python" / "dynamics" / "trans_matrix_p.npy",
    produces=BLD / "python" / "tables" / "ss_states.tex",
):
    """Compute the stochastically stable states of the perturbed process."""

    trans_matrix_p = np.load(trans_matrix_p_dir)

    ss_states = compute_stoch_stab_states(trans_matrix_p)
    ss_states.to_latex(buf=produces, index=False)


def task_save_root_edmonds_arboresence(
    states_dir=BLD / "python" / "game" / "states.npy",
    trans_matrix_un_dir=BLD / "python" / "dynamics" / "trans_matrix_un.npy",
    parameters_dir=BLD / "python" / "game" / "parameters.pkl",
    G_dir = BLD / "python" / "dynamics" / "G.net",
    produces=BLD / "python" / "dynamics" / "root.csv",
):
    """Write roots of Edmond's arboresence to a csv file."""

    with open(parameters_dir, "rb") as fp:
        parameters = pickle.load(fp)

    states = np.load(states_dir)
    trans_matrix_un = np.load(trans_matrix_un_dir)
    k = parameters["k"]
    num_act = parameters["num_act"]
    payoffs = parameters["payoffs"]
    G = nx.read_pajek(G_dir)

    rcc_index = compute_rcc_index(trans_matrix_un)
    arb = find_edmonds_arboresence(G, states, trans_matrix_un, k, num_act, payoffs)
    root_state = __find_root_state(states, arb, rcc_index)

    root_state.tofile(produces, sep=",")


def task_find_states_with_min_stoch_pot(
    states_dir=BLD / "python" / "game" / "states.npy",
    trans_matrix_un_dir=BLD / "python" / "dynamics" / "trans_matrix_un.npy",
    parameters_dir=BLD / "python" / "game" / "parameters.pkl",
    G_dir = BLD / "python" / "dynamics" / "G.net",
    produces=BLD / "python" / "dynamics" / "roots.csv",
):
    """Find all states with minimum stochastic potential through iteration."""

    with open(parameters_dir, "rb") as fp:
        parameters = pickle.load(fp)

    states = np.load(states_dir)
    trans_matrix_un = np.load(trans_matrix_un_dir)
    k = parameters["k"]
    num_act = parameters["num_act"]
    payoffs = parameters["payoffs"]
    G = nx.read_pajek(G_dir)

    roots = find_states_with_min_stoch_pot(
        G, states, trans_matrix_un, k, num_act, payoffs
    )[1]

    roots.tofile(produces, sep=",")
