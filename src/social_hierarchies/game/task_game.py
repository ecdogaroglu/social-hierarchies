"""Task for creating the state space."""

import numpy as np
import pickle

from social_hierarchies.config import BLD
from social_hierarchies.game.game import define_state_space


def task_define_state_space(
    parameters_dir=BLD / "python" / "game" / "parameters.pkl",
    produces=BLD / "python" / "game" / "states.npy",
):
    """Create the state space."""

    with open(parameters_dir, "rb") as fp:
        parameters = pickle.load(fp)

    num_act = parameters["num_act"]
    num_players = parameters["num_players"]
    m = parameters["m"]
    l = parameters["l"]


    states = define_state_space(num_act, num_players, m, l)
    np.save(produces, states)
