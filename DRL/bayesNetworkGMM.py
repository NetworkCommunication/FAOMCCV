import os
import pickle
import warnings
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

warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have valid feature names.*")

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
    saveModel(model, 'savedModel/bayes_GMMmodel.pkl')

    return sum(score) / len(score), predictions[0]

def bayesianNetVar(df):
    target_variable = "status_change"
    X, y = df.drop(columns=[target_variable]), df[target_variable]
    state_names = {column: df[column].unique().tolist() for column in df.columns}
    network = [(target_variable, x) for x in X.columns]
    # print("Network structure:", network)
    naive_bayes = BayesianNetwork(network)
    showBN(naive_bayes)

    roc_auc_value, predictions = bayesian_kfold(df, naive_bayes, roc_auc_score, state_names)


def bayesianNetWithHillCli(df):
    scores = {}  # Dictionary to store the roc_auc_score for each scoring method
    networks = {}  # Dictionary to store the network structure for each scoring method
    state_names = {column: df[column].unique().tolist() for column in df.columns}
    # for scoring in ['k2score', 'bdeuscore', 'bdsscore', 'bicscore', 'aicscore']:
    scoring = 'k2score'
    network = HillClimbSearch(df, use_cache=False).estimate(scoring_method=scoring)
    networks[scoring] = network
    model = BayesianNetwork(network)
    showBN(model)
    scores[scoring], _ = bayesian_kfold(df, model, roc_auc_score, state_names)

    pd.DataFrame(scores, index=['ROC AUC'])
    print(scores)

def bayesianGMMNetWithHillCli(dfGMM):
    scores = {}  # Dictionary to store the roc_auc_score for each scoring method
    networks = {}  # Dictionary to store the network structure for each scoring method

    use_columns = ['velocity_gmm', 'acceleration_gmm', 'lane_change', 'delta_velocity_gmm', 'deceleration_distance_gmm', 'drac_gmm', 'status_change']
    state_names = {column: dfGMM[column].unique().tolist() for column in use_columns}
    # for scoring in ['k2score', 'bdeuscore', 'bdsscore', 'bicscore', 'aicscore']:
    scoring = 'k2score'
    network = HillClimbSearch(dfGMM[use_columns], use_cache=False).estimate(scoring_method=scoring)
    networks[scoring] = network
    model = BayesianNetwork(network)
    showBN(model)
    scores[scoring], _ = bayesian_kfold(dfGMM[use_columns], model, roc_auc_score, state_names)

    pd.DataFrame(scores, index=['ROC AUC'])
    print(scores)
    return dfGMM[use_columns]

def bayesianNetWithTree(df):
    scores = {}  # Dictionary to store the roc_auc_score for each scoring method
    networks = {}  # Dictionary to store the network structure for each scoring method
    state_names = {column: df[column].unique().tolist() for column in df.columns}
    # for scoring in ['k2score', 'bdeuscore', 'bdsscore', 'bicscore', 'aicscore']:
    scoring = 'k2score'
    network = TreeSearch(df).estimate(estimator_type="chow-liu")
    networks['chow-liu'] = network
    model = BayesianNetwork(network)
    showBN(model)
    scores[scoring], _ = bayesian_kfold(df, model, roc_auc_score, state_names)

    pd.DataFrame(scores, index=['ROC AUC'])
    print(scores)

def custom_round(val):
    if isinstance(val, float):
        str_val = f"{val:.10f}"
        decimal_point_index = str_val.find('.')
        decimal_places = len(str_val) - decimal_point_index - 1
        if decimal_places > 0:
            val = round(val, 0)
        return np.float32(val)
    else:
        return val

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

def hcConst(df):
    state_names = {column: df[column].unique().tolist() for column in df.columns}
    black_list = [('drac', 'delta_velocity'),
                  ('drac', 'deceleration_distance')]
    hc_const = HillClimbSearch(df, use_cache=False).estimate(scoring_method='aicscore', black_list=black_list)
    hc_const_model = BayesianNetwork(hc_const)
    showBN(hc_const_model)

    roc_auc, _ = bayesian_kfold(df, hc_const_model, roc_auc_score, state_names)
    print(f'ROC AUC: {roc_auc:.3f}')

def saveModel(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

target = ['status_change']

def format_string(array):
    string = str(array[0])
    for item in array[1:]:
        string += f', {item}'
    return string

def exact_inference(model, variables, evidence):
    inference = VariableElimination(model)
    result = inference.query(variables=variables, evidence=evidence)
    return result

def create_dictionary(df, columns):
    dictionary = {}
    for column in columns:
        dictionary[column] = df[column].unique().tolist()
    return dictionary

def get_all_combinations(dictionary):
    if len(dictionary) == 1:
        return [{list(dictionary.keys())[0]: value} for value in dictionary[list(dictionary.keys())[0]]]
    res = []
    for k1, k2 in combinations(dictionary.keys(), 2):
        for v1, v2 in product(dictionary[k1], dictionary[k2]):
            res.append({k1: v1, k2: v2})
    return res

def he_prob_analysis(model, target, knowledge, df):
    res = pd.DataFrame(columns=knowledge + ["Prob"])
    all_values = create_dictionary(df, knowledge)
    all_queries = get_all_combinations(all_values)

    for query in all_queries:
        result = exact_inference(model, target, query)
        query["Prob"] = result.values[1]
        res.loc[len(res)] = query

    return res.sort_values(by=knowledge[0], ascending=False).reset_index(drop=True)

def prediction(df, model, query):
    target = ['status_change']
    labels = df.columns.tolist()
    labels.remove('status_change')
    variables = []

    for label in labels:
        if label not in query.keys():
            variables.append(label)

    base_result = exact_inference(model, target, query)
    probs = base_result.values
    probs = np.round(probs * 100, 2)

    my_dict = {}
    for col in df.drop(target, axis=1).columns:
        my_dict[col] = df[col].unique().tolist()
    exam_df = pd.DataFrame(columns=['exam', 'outcome', 'prob'])

    for var in variables:
        for val in my_dict[var]:
            query[var] = val
            result = exact_inference(model, target, query)
            exam_df.loc[len(exam_df)] = [var, val, round(result.values[1], 2)]
            del query[var]

    exam_df.sort_values(by='prob', ascending=False, inplace=True)

    return probs[1]

def gaussianProcessing(df):
    use_columns = ['velocity_gmm', 'acceleration_gmm', 'lane_change', 'delta_velocity_gmm', 'deceleration_distance_gmm',
                   'drac_gmm', 'status_change']
    gmm_velocity = GaussianMixture(n_components=9, random_state=1).fit(df[['velocity']])
    gmm_acceleration = GaussianMixture(n_components=9, random_state=1).fit(df[['acceleration']])
    gmm_delta_velocity = GaussianMixture(n_components=9, random_state=1).fit(df[['delta_velocity']])
    gmm_delta_deceleration_distance = GaussianMixture(n_components=8, random_state=1).fit(df[['deceleration_distance']])
    gmm_drac = GaussianMixture(n_components=9, random_state=1).fit(df[['drac']])

    df['velocity_gmm'] = gmm_velocity.predict(df[['velocity']])
    df['acceleration_gmm'] = gmm_acceleration.predict(df[['acceleration']])
    df['delta_velocity_gmm'] = gmm_delta_velocity.predict(df[['delta_velocity']])
    df['deceleration_distance_gmm'] = gmm_delta_deceleration_distance.predict(df[['deceleration_distance']])
    df['drac_gmm'] = gmm_drac.predict(df[['drac']])
    return df[use_columns], (gmm_velocity, gmm_acceleration, gmm_delta_velocity, gmm_delta_deceleration_distance, gmm_drac)

if __name__ == '__main__':
    columns = ['velocity', 'acceleration', 'lane_change', 'delta_velocity', 'deceleration_distance', 'drac', 'status_change']
    df = pd.read_csv('dataPreprocess/bayesianDataset5000.csv')[columns]
    # df['status_change'] = df['status_change'].astype('int64')

    # bayesianNetVar(df)


    # bayesianNetWithHillCli(df)
    # bayesianNetWithTree(df)

    dfGMM, _ = gaussianProcessing(df)
    # bayesianGMMNetWithHillCli(dfGMM)


    model = load_model('savedModel/bayes_GMMmodel.pkl')
    # evidences = ["lane_change"]
    # result = he_prob_analysis(model, target, evidences, df)
    # print(result)

    # Practical use of the model
    # model = load_model('savedModel/bayes_GMMmodel.pkl')
    # print(model.nodes())
    # query = {'velocity': 6.3, 'acceleration': -11.2, 'lane_change': 1, 'delta_velocity': -1.97, 'deceleration_distance': 0.7389999999999901, 'drac': 5.251556156968946}
    query = {'velocity': 6.3, 'acceleration': -11.2, 'lane_change': 1, 'delta_velocity': -1.7,
             'deceleration_distance': 0.7, 'drac': 5.251556}

    _, gmm = gaussianProcessing(df)
    query['velocity_gmm'] = gmm[0].predict([[query['velocity']]])[0]
    query['acceleration_gmm'] = gmm[1].predict([[query['acceleration']]])[0]
    query['delta_velocity_gmm'] = gmm[2].predict([[query['delta_velocity']]])[0]
    query['deceleration_distance_gmm'] = gmm[3].predict([[query['deceleration_distance']]])[0]
    query['drac_gmm'] = gmm[4].predict([[query['drac']]])[0]

    query.pop('velocity')
    query.pop('acceleration')
    query.pop('delta_velocity')
    query.pop('deceleration_distance')
    query.pop('drac')
    #
    #
    prediction(dfGMM, model, query)



