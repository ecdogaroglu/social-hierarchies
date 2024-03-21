"""Tasks running the results formatting (tables, figures)."""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle


from social_hierarchies.config import BLD
from social_hierarchies.dynamics.graph import find_states_with_min_stoch_pot, ___index_to_state
from social_hierarchies.final.plot import (
    plot_stationary_distr,
    plot_rcc_graph,
    plot_edmonds_arboresence,
)


def task_plot_stationary_distr(
    states_dir=BLD / "python" / "game" / "states.npy",
    trans_matrix_p_dir=BLD / "python" / "dynamics" / "trans_matrix_p.npy",
    produces=BLD / "python" / "figures" / "s_distr.png",
):
    """Plot the stationary distribution of the perturbed process."""

    states = np.load(states_dir)
    trans_matrix_p = np.load(trans_matrix_p_dir)

    x, s_distr = plot_stationary_distr(states, trans_matrix_p)

    plt.figure(2)
    plt.plot(x, s_distr, color="k")
    plt.savefig(produces)


def task_plot_edmonds_arboresence(
    states_dir=BLD / "python" / "game" / "states.npy",
    trans_matrix_un_dir=BLD / "python" / "dynamics" / "trans_matrix_un.npy",
    parameters_dir=BLD / "python" / "game" / "parameters.pkl",
    produces=BLD / "python" / "figures" / "edmonds_arboresence.png",
):
    """Plot the Edmond's arboresence."""

    with open(parameters_dir, "rb") as fp:
        parameters = pickle.load(fp)

    states = np.load(states_dir)
    trans_matrix_un = np.load(trans_matrix_un_dir)
    k = parameters["k"]
    num_act = parameters["num_act"]
    payoffs = parameters["payoffs"]

    arb, labels, edge_labels = plot_edmonds_arboresence(
        states, k, num_act, payoffs, trans_matrix_un
    )

    plt.figure(0)
    pos = nx.spring_layout(arb, seed=7)
    nx.draw(
        arb,
        pos=pos,
        with_labels=True,
        labels=labels,
        font_weight="normal",
        width=0.5,
        arrows=True,
        arrowsize=30,
        margins=0.2,
        node_color="white",
        font_size=8,
        node_size=1000,
        connectionstyle="arc3, rad = 0.1",
    )
    nx.draw_networkx_edge_labels(arb, pos=pos, edge_labels=edge_labels)
    plt.savefig(produces)


def task_plot_G_rcc(
    states_dir=BLD / "python" / "game" / "states.npy",
    trans_matrix_un_dir=BLD / "python" / "dynamics" / "trans_matrix_un.npy",
    parameters_dir=BLD / "python" / "game" / "parameters.pkl",
    produces=BLD / "python" / "figures" / "G_rcc.png",
):
    """Plot the RCC graph."""

    with open(parameters_dir, "rb") as fp:
        parameters = pickle.load(fp)

    states = np.load(states_dir)
    trans_matrix_un = np.load(trans_matrix_un_dir)
    k = parameters["k"]
    num_act = parameters["num_act"]
    payoffs = parameters["payoffs"]

    G_rcc, labels, edge_labels = plot_rcc_graph(
        states, k, num_act, payoffs, trans_matrix_un
    )

    plt.figure(1)
    pos = nx.spring_layout(G_rcc, seed=7)
    nx.draw(
        G_rcc,
        with_labels=True,
        pos=pos,
        labels=labels,
        font_weight="normal",
        width=0.5,
        arrows=True,
        arrowsize=30,
        margins=0.2,
        node_color="white",
        font_size=8,
        node_size=1000,
        connectionstyle="arc3, rad = 0.3",
    )
    nx.draw_networkx_edge_labels(G_rcc, pos=pos, edge_labels=edge_labels, font_size=8)
    plt.savefig(produces)


def task_plot_multiple_min_arbs(
    states_dir=BLD / "python" / "game" / "states.npy",
    trans_matrix_un_dir=BLD / "python" / "dynamics" / "trans_matrix_un.npy",
    parameters_dir=BLD / "python" / "game" / "parameters.pkl",
    produces=BLD / "python" / "figures" / "num_arbs.npy",
):
    """Plot the multiple minimum arboresences."""

    with open(parameters_dir, "rb") as fp:
        parameters = pickle.load(fp)

    states = np.load(states_dir)
    trans_matrix_un = np.load(trans_matrix_un_dir)
    k = parameters["k"]
    num_act = parameters["num_act"]
    payoffs = parameters["payoffs"]

    arbs = find_states_with_min_stoch_pot(states, trans_matrix_un, k, num_act, payoffs)[
        0
    ]

    for i in range(len(arbs)):
        plt.figure(10 + i)
        indices = list(arbs[i].nodes())
        labels = {}
        for j in indices:
            labels[j] = states[j] + np.ones((states.shape[1], states.shape[2]))

        edge_labels = nx.get_edge_attributes(arbs[i], "weight")
        pos = nx.spring_layout(arbs[i], seed=7)

        nx.draw(
            arbs[i],
            pos=pos,
            with_labels=True,
            labels=labels,
            font_weight="normal",
            width=0.5,
            arrows=True,
            arrowsize=30,
            margins=0.2,
            node_color="white",
            font_size=8,
            node_size=1000,
            connectionstyle="arc3, rad = 0.1",
        )
        nx.draw_networkx_edge_labels(arbs[i], pos=pos, edge_labels=edge_labels)
        plt.savefig(BLD / "python" / "figures" / f"arb_{i}.png")

    num_arbs = np.array(len(arbs))

    np.save(produces, num_arbs)
