"""Functions creating markov chains and extracting their properties."""

import numpy as np
import quantecon as qe
import pandas as pd

from social_hierarchies.game.game import _best_response_prob



# What's the probability that player i's position in jth level will be the action s in the next period?

def pl_pos_is_s(pl,level,s,payoffs,state):
    if level != 0 and level != (state.shape[0]-1): # inner levels
        pl_demoted = _prob_player_is("demote", payoffs,state[level])[pl] * _prob_trans_is_action("promote", payoffs,state[level-1])[s]
        pl_promoted = _prob_player_is("promote", payoffs,state[level])[pl] * _prob_trans_is_action("demote", payoffs,state[level+1])[s]

        prob_s = pl_demoted + pl_promoted

    if level == 0: # bottom level
        pl_kept = _prob_player_is_kept("promote", payoffs, state[level])[pl] * _prob_s_cond_on_kept("promote", pl, payoffs, state[level])[s]
        pl_promoted = _prob_player_is("promote", payoffs,state[level])[pl] * _prob_trans_is_action("demote", payoffs,state[level+1])[s]

        prob_s = pl_kept + pl_promoted

    if level == (state.shape[0]-1): # top level
        pl_demoted = _prob_player_is("demote", payoffs,state[level])[pl] * _prob_trans_is_action("promote", payoffs,state[level-1])[s]
        pl_kept = _prob_player_is_kept("demote", payoffs, state[level])[pl] * _prob_s_cond_on_kept("demote", pl, payoffs, state[level])[s]

        prob_s = pl_demoted + pl_kept

    return prob_s




# What's the probability that a given transition will be someone who played the action s?
def _prob_trans_is_action(transition, payoffs,hist):
    probs = []
    for i in range(payoffs.shape[0]):
        probs.append(__prob_trans_is_s_given_hist(transition, i,payoffs,hist))
    return probs

def __prob_trans_is_s_given_hist(transition, s,payoffs,hist):
    to_sum=___prob_outcomes_trans_action_is_s(transition, payoffs,s) * ___prob_outcome_happening_given_hist(payoffs,hist)

    return np.sum(to_sum)

def ___prob_outcome_happening_given_hist (payoffs, hist):
    prob = []
    for i in range(payoffs.shape[0]):
        for j in range(payoffs.shape[1]):
            prob.append(_best_response_prob(hist,i,0,k,num_act,payoffs) * _best_response_prob(hist,j,1,k,num_act,payoffs))
    prob =np.array(prob)
    return prob

def ___prob_outcomes_trans_action_is_s(transition, payoffs,s):
    prob = []
    for i in range(payoffs.shape[0]):
        for j in range(payoffs.shape[1]):
            if s in ____trans_cand_s_of_outcome(transition, payoffs,i,j):
                prob.append(1/np.unique(____trans_cand_s_of_outcome(transition, payoffs,i,j)).shape[0])
            else:
                prob.append(0)   
    prob =np.array(prob)

    return prob

def ____trans_cand_s_of_outcome(transition, payoffs, i,j):
    if transition == "promote":
        cand_players =np.flatnonzero(payoffs[i,j] == np.max(payoffs[i,j]))

    elif transition == "demote":
        cand_players =np.flatnonzero(payoffs[i,j] == np.min(payoffs[i,j]))

    strategies_played = [i,j]
    strategies_promoted = []
    for p in list(cand_players):
        strategies_promoted.append(strategies_played[p])
    strategies_promoted = np.array(strategies_promoted)
    return strategies_promoted



# What's the probability that a kept player played s?

def _prob_s_cond_on_kept(transition, pl, payoffs,hist):
    matrix =(___prob_outcomes_kept_player_is_pl(transition, payoffs,pl) * ___prob_outcome_happening_given_hist_np(payoffs,hist))/_prob_player_is_kept(transition, payoffs,hist)[pl]
    
    return np.sum(np.nan_to_num(matrix), axis=1)

def _prob_player_is_kept(transition, payoffs,hist):
    probs = []
    for i in range(payoffs.shape[0]):
        probs.append(__prob_kept_is_pl_given_hist(transition, i,payoffs,hist))
    return probs

def __prob_kept_is_pl_given_hist(transition, pl,payoffs,hist): 
    to_sum=___prob_outcomes_kept_player_is_pl(transition, payoffs,pl) * ___prob_outcome_happening_given_hist_np(payoffs,hist)

    return np.sum(to_sum)

def ___prob_outcome_happening_given_hist_np (payoffs, hist):
    prob = np.zeros((num_act,num_act))
    for i in range(payoffs.shape[0]):
        for j in range(payoffs.shape[1]):
            prob[i,j] = _best_response_prob(hist,i,0,k,num_act,payoffs) * _best_response_prob(hist,j,1,k,num_act,payoffs)
    return prob

def ___prob_outcomes_kept_player_is_pl(transition, payoffs, pl): # her outcome için kalma ihtimalim
    prob = np.zeros((num_act,num_act))
    for i in range(payoffs.shape[0]):
        for j in range(payoffs.shape[1]):
            if pl not in ____trans_cand_pl_of_outcome(transition, payoffs,i,j):
                prob[i,j] = 1 
            elif pl in ____trans_cand_pl_of_outcome(transition, payoffs,i,j) and ____trans_cand_pl_of_outcome(transition, payoffs,i,j).shape[0] == 2:
                prob[i,j] = 0.5
            elif pl in ____trans_cand_pl_of_outcome(transition, payoffs,i,j) and ____trans_cand_pl_of_outcome(transition, payoffs,i,j).shape[0] == 1:
                prob[i,j] = 0

    return prob

def ____trans_cand_pl_of_outcome(transition, payoffs, i,j):
    if transition == "promote":
        cand_players =np.flatnonzero(payoffs[i,j] == np.max(payoffs[i,j]))

    elif transition == "demote":
        cand_players =np.flatnonzero(payoffs[i,j] == np.min(payoffs[i,j]))

    return cand_players



# What's the probability that a player transitions?

def _prob_player_is(transition, payoffs,hist):
    # A list describing the probability that each player will be transitioned for a given history

    probs = []
    for i in range(payoffs.shape[2]):
        probs.append(__prob_trans_is_pl_given_hist(transition, i,payoffs,hist))
    return probs

def __prob_trans_is_pl_given_hist(transition, pl,payoffs,hist):
    # A number describing the probability that a certain player will be transitioned for a given history

    to_sum=___prob_outcomes_trans_player_is_pl(transition, payoffs,pl) * ___prob_outcome_happening_given_hist(payoffs,hist)

    return np.sum(to_sum)

def ___prob_outcome_happening_given_hist (payoffs, hist):
    prob = []
    for i in range(payoffs.shape[0]):
        for j in range(payoffs.shape[1]):
            prob.append(_best_response_prob(hist,i,0,k,num_act,payoffs) * _best_response_prob(hist,j,1,k,num_act,payoffs))
    prob =np.array(prob)
    return prob

def ___prob_outcomes_trans_player_is_pl(transition, payoffs,pl):
    # An array describing the probability that a certain player will be transitioned for each outcome

    prob = []
    for i in range(payoffs.shape[0]):
        for j in range(payoffs.shape[1]):
            if pl in ____trans_cand_pl_of_outcome(transition, payoffs,i,j):
                prob.append(1/np.unique(____trans_cand_pl_of_outcome(transition, payoffs,i,j)).shape[0])
            else:
                prob.append(0)   
    prob =np.array(prob)

    return prob



def pl_pos_is_s_by_mistake(pl,level,s,payoffs,state):
    if level != 0 and level != (state.shape[0]-1): # inner levels
        pl_demoted = 0.5 * _prob_trans_is_action_perturbed("promote", payoffs,state[level-1])[s]
        pl_promoted = 0.5 * _prob_trans_is_action_perturbed("demote", payoffs,state[level+1])[s]

        prob_s = pl_demoted + pl_promoted

    if level == 0: # bottom level
        pl_kept = 0.5 * 0.5
        pl_promoted = 0.5 * _prob_trans_is_action_perturbed("demote", payoffs,state[level+1])[s]

        prob_s = pl_kept + pl_promoted

    if level == (state.shape[0]-1): # top level
        pl_demoted = 0.5 * _prob_trans_is_action_perturbed("promote", payoffs,state[level-1])[s]
        pl_kept = 0.5 * 0.5

        prob_s = pl_demoted + pl_kept

    return prob_s


def _prob_trans_is_action_perturbed(transition, payoffs,hist):
    probs = []
    for i in range(payoffs.shape[0]):
        probs.append(__prob_trans_is_s_given_hist_perturbed(transition, i,payoffs,hist))
    return probs

def __prob_trans_is_s_given_hist_perturbed(transition, s,payoffs,hist):
    to_sum=___prob_outcomes_trans_action_is_s_perturbed(transition, payoffs,s) * ___prob_outcome_happening_given_hist_perturbed(payoffs,hist)

    return np.sum(to_sum)

def ___prob_outcome_happening_given_hist_perturbed (payoffs, hist):
    prob = []
    for i in range(payoffs.shape[0]):
        for j in range(payoffs.shape[1]):
            prob.append(1/(payoffs.shape[0]*payoffs.shape[1]))
    prob =np.array(prob)
    return prob

def ___prob_outcomes_trans_action_is_s_perturbed(transition, payoffs,s):
    prob = []
    for i in range(payoffs.shape[0]):
        for j in range(payoffs.shape[1]):
            if s in [i,j]:
                prob.append(1/np.unique(np.array([i,j])).shape[0])
            else:
                prob.append(0)   
    prob =np.array(prob)

    return prob