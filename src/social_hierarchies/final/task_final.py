"""Tasks running the results formatting (tables, figures)."""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pytask

from social_hierarchies.config import BLD
from social_hierarchies.dynamics.graph import find_states_with_min_stoch_pot, ___index_to_state, __find_root_index, compute_rcc_index
from social_hierarchies.final.plot import (
    plot_stationary_distr,
    plot_rcc_graph,
    plot_edmonds_arboresence,
    plot_shortest_path_example,
)


def task_plot_stationary_distr_unperturbed(
    states_dir=BLD / "python" / "game" / "states.npy",
    trans_matrix_un_dir=BLD / "python" / "dynamics" / "trans_matrix_un.npy",
    produces=BLD / "python" / "figures" / "s_distr_un_0.png",
):
    """Plot the stationary distributions of the unperturbed process."""

    states = np.load(states_dir)
    trans_matrix_un = np.load(trans_matrix_un_dir)

    x, s_distr_0 = plot_stationary_distr(states, trans_matrix_un, index=0)

    plt.figure(0)
    plt.plot(x, s_distr_0, color="k")
    plt.savefig(produces)

    x, s_distr_1 = plot_stationary_distr(states, trans_matrix_un, index=1)

    plt.figure(1)
    plt.plot(x, s_distr_1, color="k")
    plt.savefig(BLD / "python" / "figures" / "s_distr_un_1.png")

    x, s_distr_2 = plot_stationary_distr(states, trans_matrix_un, index=2)

    plt.figure(2)
    plt.plot(x, s_distr_2, color="k")
    plt.savefig(BLD / "python" / "figures" / "s_distr_un_2.png")


def task_plot_stationary_distr_perturbed(
    states_dir=BLD / "python" / "game" / "states.npy",
    trans_matrix_p_dir=BLD / "python" / "dynamics" / "trans_matrix_p.npy",
    produces=BLD / "python" / "figures" / "s_distr_p.png",
):
    """Plot the stationary distribution of the perturbed process."""

    states = np.load(states_dir)
    trans_matrix_p = np.load(trans_matrix_p_dir)

    x, s_distr_p = plot_stationary_distr(states, trans_matrix_p)

    plt.figure(5)
    plt.plot(x, s_distr_p, color="k")
    plt.savefig(produces)

@pytask.mark.try_last
def task_plot_G_rcc(
    states_dir=BLD / "python" / "game" / "states.npy",
    trans_matrix_un_dir=BLD / "python" / "dynamics" / "trans_matrix_un.npy",
    G_dir = BLD / "python" / "dynamics" / "G.net",
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
    G = nx.read_pajek(G_dir)

    G_rcc, labels, edge_labels = plot_rcc_graph(
        G, states, k, num_act, payoffs, trans_matrix_un
    )

    plt.figure(6)
    pos = nx.spring_layout(G_rcc, seed=7)
    nx.draw(G_rcc, pos=pos, with_labels=True, labels=labels, node_size=10000, margins=0.2, font_size=12, font_weight="bold", arrowsize=30)
    nx.draw_networkx_edge_labels(G_rcc, pos, edge_labels=edge_labels, label_pos=0.5, font_size=12)

    plt.savefig(produces)

@pytask.mark.try_last
def task_plot_G_sp_example(
    states_dir=BLD / "python" / "game" / "states.npy",
    parameters_dir=BLD / "python" / "game" / "parameters.pkl",
    G_dir = BLD / "python" / "dynamics" / "G.net",
    produces=BLD / "python" / "figures" / "G_sp_ex.png",
):
    """Plot the shortest path between two recurrent communication classes."""

    with open(parameters_dir, "rb") as fp:
        parameters = pickle.load(fp)

    states = np.load(states_dir)
    k = parameters["k"]
    num_act = parameters["num_act"]
    payoffs = parameters["payoffs"]
    G = nx.read_pajek(G_dir)


    G_sp_ex, sp_labels, sp_edge_labels = plot_shortest_path_example(
        G, states, k, num_act, payoffs
    )

    plt.figure(7)
    pos= nx.spring_layout(G_sp_ex, k=4, seed=23)
    colors = ["red", "#1f78b4", "#1f78b4", "#1f78b4", "#1f78b4", "red"]
    nx.draw(G_sp_ex, pos=pos, with_labels=True, labels=sp_labels, node_size=6000, margins=0.1, font_size=10, font_weight="bold",node_color=colors, arrowsize=15)
    nx.draw_networkx_edge_labels(G_sp_ex, pos, edge_labels=sp_edge_labels, label_pos=0.5, font_size=12)
    plt.savefig(produces)

@pytask.mark.try_last
def task_plot_edmonds_arboresence(
    states_dir=BLD / "python" / "game" / "states.npy",
    trans_matrix_un_dir=BLD / "python" / "dynamics" / "trans_matrix_un.npy",
    G_dir = BLD / "python" / "dynamics" / "G.net",
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
    G = nx.read_pajek(G_dir)


    arb, labels, edge_labels = plot_edmonds_arboresence(
        G, states, k, num_act, payoffs, trans_matrix_un
    )

    plt.figure(8)
    pos = nx.spring_layout(arb, seed=7)

    colors = ["#1f78b4","#1f78b4", "red",]
    nx.draw(arb, pos=pos, with_labels=True, labels=labels, node_size=10000, margins=0.2, font_size=12, font_weight="bold", node_color=colors, arrowsize=30)
    nx.draw_networkx_edge_labels(arb, pos, edge_labels=edge_labels, label_pos=0.5, font_size=12)

    plt.savefig(produces)


@pytask.mark.try_last
def task_plot_multiple_min_arbs(
    states_dir=BLD / "python" / "game" / "states.npy",
    trans_matrix_un_dir=BLD / "python" / "dynamics" / "trans_matrix_un.npy",
    G_dir = BLD / "python" / "dynamics" / "G.net",
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
    rcc_index = compute_rcc_index(trans_matrix_un)

    G = nx.read_pajek(G_dir)

    arbs = find_states_with_min_stoch_pot(G, states, trans_matrix_un, k, num_act, payoffs)[
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

        root = __find_root_index(arbs[i],rcc_index)
        
        colors = []
        for index in indices:
            if index in root:
                colors.append("red")
            else:
                colors.append("#1f78b4")
        #colors = ["red", "#1f78b4", "#1f78b4"]
        nx.draw(arbs[i], pos=pos, with_labels=True, labels=labels, node_size=10000, margins=0.2, font_size=12, font_weight="bold", node_color=colors, arrowsize=30)
        nx.draw_networkx_edge_labels(arbs[i], pos, edge_labels=edge_labels, label_pos=0.5, font_size=14)
        plt.savefig(BLD / "python" / "figures" / f"arb_{i}.png")

    num_arbs = np.array(len(arbs))

    np.save(produces, num_arbs)
