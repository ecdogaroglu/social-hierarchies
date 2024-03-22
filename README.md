# Social Hierarchies with Strategic Ability


[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ecdogaroglu/social_hierarchies/main.svg)](https://results.pre-commit.ci/latest/github/ecdogaroglu/social_hierarchies/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Abstract

This paper explores the dynamics of social hierarchies, where agents possess strategic decision making abilities. Such abilities allow the agents to "have a say" on their lifetime destiny by responding optimally to the past play, while they are still subject to limitations of their social environment.
I deploy the adaptive of Young (1993) in a social hierarchy
structure, give a novel characterization of social mobility dynamics in a framework with sophisticated
decision making abilities and model the resulting stochastic process as a Markov chain.
Then I employ several perturbations of the stochastic process and use graph theory to explore 
the asymptotic behavior of this social system.

See social_hierarchies.pdf for the related research paper including the theory developed and a discussion of the computational results.

## Usage

To get started, create and activate the environment with

```console
$ conda/mamba env create
$ conda activate social_hierarchies
```

To build the project, type

```console
$ pytask
```
Runtime is aroun 10 minutes with the default configuration.


## Modules

### task_parameters.py

This is where the parametrization takes place and stored temporarily for the global acccess throughout the project.

### game.py

This module handles the creation of the state space from the given parameters and computes best response probabilities
that are fundamental for the characterization of the stochastic processes.


## probabilities.py

This module characterizes the social mobility dynamics by computing the corresponding transition probabilities with the help of best response probabilities.

### markov_chain.py

Through transition probabilities, transition matrices of the two Markov chains and their
important properties like recurrent communication classes and the stationary distribution are calculated.

### graph.py

Through utilization of best response probabilities, weights of two directed graphs are calculated. While the first graph takes states of the stochastic process as vertices, the second one takes the recurrent communication classes of the unperturbed process as vertices. The graphs allow for the utilization of the shortest path and the optimum arboresence algorithms.

### plot.py

The stationary distribution, graph of recurrent communication classes, shortest path and minimum arboresences are plotted for a better visual understanding.

### social_hiearchies.tex

Analyse the research results to be complied into a pdf file.

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).


## References

Young, H. Peyton. "The evolution of conventions." Econometrica: Journal of the Econometric Society (1993): 57-84.
