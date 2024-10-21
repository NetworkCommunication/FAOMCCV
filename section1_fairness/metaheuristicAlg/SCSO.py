import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, StructureEstimator
from pgmpy.sampling import BayesianModelSampling
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from pgmpy.estimators import StructureScore, HillClimbSearch, BicScore
from collections import deque
from itertools import permutations
from pgmpy.base import DAG


class SCSOHillClimbSearch(StructureEstimator):
    # epochs=10000, pop_size=100
    def __init__(self, data, epochs=100, pop_size=100, **kwargs):
        super().__init__(data, **kwargs)
        self.epochs = epochs
        self.pop_size = pop_size
        self.population = [DAG() for _ in range(pop_size)]
        for dag in self.population:
            dag.add_nodes_from(self.variables)

    def score_population(self, population, scoring_method):
        scores = []
        score_fn = scoring_method.local_score
        for model in population:
            score = sum(score_fn(node, model.predecessors(node)) for node in model.nodes())
            scores.append(score)
        return scores

    def evolve(self, scoring_method):
        score_fn = scoring_method.local_score
        for epoch in range(self.epochs):
            new_population = []
            scores = self.score_population(self.population, scoring_method)
            best_idx = np.argmax(scores)
            best_dag = self.population[best_idx]

            for _ in range(self.pop_size):
                new_dag = DAG()
                new_dag.add_nodes_from(self.variables)
                for node in self.variables:
                    if np.random.random() < 0.5:  # Randomly choosing whether to add an edge
                        potential_parents = set(self.variables) - {node}
                        selected_parent = np.random.choice(list(potential_parents))
                        if not nx.has_path(new_dag, node, selected_parent):
                            new_dag.add_edge(selected_parent, node)
                new_population.append(new_dag)

            self.population = new_population

        return best_dag

    def estimate(self, scoring_method="aicscore"):
        if isinstance(scoring_method, str):
            score_class = {"aicscore": BicScore}[scoring_method]
            scoring_instance = score_class(self.data)
        else:
            scoring_instance = scoring_method

        best_model = self.evolve(scoring_instance)
        return best_model