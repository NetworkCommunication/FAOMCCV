import os
import pickle
from itertools import combinations, product

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pgmpy.estimators import ExpectationMaximization, MaximumLikelihoodEstimator, BayesianEstimator, HillClimbSearch, \
    TreeSearch
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFWriter
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, KFold
from pgmpy.models import BayesianNetwork
from tensorboard.notebook import display

from bayesianNetwork.metaheuristicAlg.SCSO import SCSOHillClimbSearch


def bayesian_kfold(df, model, metric, state_names):
    target_variable = "status_change"
    score = []
    predictions = []
    kfold = KFold(5, shuffle=True, random_state=1)
    for train, test in kfold.split(df):
        model.cpds = []
        model.fit(df.iloc[train, :], estimator=BayesianEstimator, state_names=state_names, prior_type='dirichlet', pseudo_counts=0.1)
        y_pred = model.predict(df.drop(columns=target_variable, axis=1).iloc[test, :])
        score.append(
            metric(df[target_variable].iloc[test], y_pred[target_variable]))
        predictions.append(y_pred)

    return sum(score) / len(score), predictions[0]

def bayesianNetWithHillCli(df):
    scores = {}  # Dictionary to store the roc_auc_score for each scoring method
    networks = {}  # Dictionary to store the network structure for each scoring method
    state_names = {column: df[column].unique().tolist() for column in df.columns}
    # for scoring in ['k2score', 'bdeuscore', 'bdsscore', 'bicscore', 'aicscore']:
    scoring = 'aicscore'
    network = SCSOHillClimbSearch(df).estimate(scoring_method=scoring)
    networks[scoring] = network
    model = BayesianNetwork(network)
    showBN(model)
    scores[scoring], _ = bayesian_kfold(df, model, roc_auc_score, state_names)

    pd.DataFrame(scores, index=['ROC AUC'])
    print(scores)

def bayesianGMMNetWithSCSO(dfGMM):
    scores = {}  # Dictionary to store the roc_auc_score for each scoring method
    networks = {}  # Dictionary to store the network structure for each scoring method
    use_columns = ['velocity_gmm', 'acceleration_gmm', 'lane_change', 'delta_velocity_gmm', 'deceleration_distance_gmm', 'drac_gmm', 'status_change']
    state_names = {column: dfGMM[column].unique().tolist() for column in use_columns}
    # for scoring in ['k2score', 'bdeuscore', 'bdsscore', 'bicscore', 'aicscore']:
    scoring = 'bicscore'
    network = SCSOHillClimbSearch(dfGMM[use_columns]).estimate(scoring_method=scoring)
    networks[scoring] = network
    model = BayesianNetwork(network)
    showBN(model)
    scores[scoring], _ = bayesian_kfold(dfGMM[use_columns], model, roc_auc_score, state_names)

    pd.DataFrame(scores, index=['ROC AUC'])
    print(scores)
    return dfGMM[use_columns]

def showBN(model, show=True):
    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    # plt.figure(figsize=(8, 6))  # Optional: define figure size
    ax = plt.gca()  # Get current axes instance
    pos = nx.layout.circular_layout(G)  # or use spring_layout, shell_layout for different layouts
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold',
            arrowstyle='-|>', arrowsize=12)
    plt.title('Bayesian Network')
    if show:
        plt.show()

def gaussianProcessing(df):
    use_columns = ['velocity_gmm', 'acceleration_gmm', 'lane_change', 'delta_velocity_gmm', 'deceleration_distance_gmm',
                   'drac_gmm', 'status_change']
    gmm_velocity = GaussianMixture(n_components=6, random_state=1).fit(df[['velocity']])
    gmm_acceleration = GaussianMixture(n_components=5, random_state=1).fit(df[['acceleration']])
    gmm_delta_velocity = GaussianMixture(n_components=4, random_state=1).fit(df[['delta_velocity']])
    gmm_delta_deceleration_distance = GaussianMixture(n_components=8, random_state=1).fit(df[['deceleration_distance']])
    gmm_drac = GaussianMixture(n_components=6, random_state=1).fit(df[['drac']])

    df['velocity_gmm'] = gmm_velocity.predict(df[['velocity']])
    df['acceleration_gmm'] = gmm_acceleration.predict(df[['acceleration']])
    df['delta_velocity_gmm'] = gmm_delta_velocity.predict(df[['delta_velocity']])
    df['deceleration_distance_gmm'] = gmm_delta_deceleration_distance.predict(df[['deceleration_distance']])
    df['drac_gmm'] = gmm_drac.predict(df[['drac']])
    return df[use_columns], (gmm_velocity, gmm_acceleration, gmm_delta_velocity, gmm_delta_deceleration_distance, gmm_drac)

if __name__ == '__main__':
    columns = ['velocity', 'acceleration', 'lane_change', 'delta_velocity', 'deceleration_distance', 'drac', 'status_change']
    df = pd.read_csv('../filtered_I80_data.csv')[columns]

    dfGMM, _ = gaussianProcessing(df)
    bayesianGMMNetWithSCSO(dfGMM)