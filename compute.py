import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import time
import random
from tqdm import tqdm

import argparse

plt.style.use('ggplot')

import multiprocessing as mp
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Path of the dataset.')
parser.add_argument('--n_cores', type=int, help='Number of CPU cores to use.')
parser.add_argument('--test_size',type=float, help='Size of the test fold in float')
parser.add_argument('--heur', type=str, choices=['RA','JA','AA','PA'], help='Heuristic to use')
parser.add_argument('--plots', type=str, choices=['YES','NO'], help='Display the plots? Default is just ROC curve')
parser.add_argument('--confusion_mat', type=str, choices=['YES','NO'], help='Whether to compute Confusion Matrix')
parser.add_argument('--thresh_bin', type=float, help='Threshold which binarises the list of scores.')

args = parser.parse_args()

flag = True

if args.n_cores >= mp.cpu_count() or args.n_cores == 0:
    print('Invalid number of cores.')
    flag = False
if 0.1 < args.test_size > 0.99:
    print('Invalid proportion of the test fold.')
    flag = False
if 0.1 < args.thresh_bin > 0.99:
    print('Invalid value of the binary threshold.')
    flag = False

if flag == False:
    exit()

print('\nStarting the algorithm...')
print('Parameters:',' '.join(f'{k}={v}' for k, v in vars(args).items()))

# Load a graph into networkX object.
try:
    Graph = nx.read_gpickle(args.dataset)
    print('Dataset loaded.')
except:
    print('Invalid dataset path. Stopping...')
t1 = time.time()

# Fraction of edges you want to remove from the training dataset. Imagine it being like train-test split.
proportion_edges = args.test_size

# Select this fraction of edges from the main graph by sampling from the graph.
edge_subset = random.sample(Graph.edges(), int(proportion_edges * Graph.number_of_edges()))
print('Testing split done.')
train = Graph.copy()

# Remove these edges from the dataset and thus create the training split.
train.remove_edges_from(edge_subset)
print('Training split done.')

# Return non-existent edges from the graph. Convert it to a Pythonic list from NetworkX iterator.
print('\nComputing the list of non-edges.')
list_of_non_edges = nx.non_edges(train)
non_edges = []
for start,end in tqdm(list_of_non_edges):
    non_edges.append((start,end))
    
# Transform the list of non_edges to a dictionary to reduce the algorithm's complexity from O**2 to O,
# as lookup in dictionary is O(1) rather than searching a list O(O**2).
print('\nCasting the test fold to dictionary.')
edge_subset_dict = {}
for x, y in tqdm(edge_subset):
    edge_subset_dict.setdefault(x, []).append(y)

# Split the array into N_CORES_TO_USE splits which will be placed into N_CORES_TO_USE cores.
split_arrays = np.array_split(np.array(non_edges), args.n_cores)

print('\nNumber of distinct nodes and edges', train.number_of_nodes(), train.number_of_edges())
print('Number of non existent edges in the graph is', len(non_edges))

# Create multiprocessing-specific structure 'Manager' and 'return_list'
manager = mp.Manager()
return_list = manager.list()

# The algorithm was tested on the following 4 heuristics of Networkx library.
# resource_allocation_index, jaccard_coefficient, adamic_adar_index, preferential_attachment
heuristics = {'RA':nx.resource_allocation_index,
              'JA':nx.jaccard_coefficient,
              'AA':nx.adamic_adar_index,
              'PA':nx.preferential_attachment}

chosen_heur = heuristics[args.heur]

# Function which is executed by each core on each of the splits
# Calculate the predictions using the heuristic
# and evaluate if the predicted key-value pairs are in the test split.
def predict_get_scores(Graph, heur, split, return_list, edge_subset):
    predictions = heur(Graph, split)    
    scores, labels = [],[]
    for (start,end,value) in tqdm(predictions):
        try:
            # Check if the predicted key-value pair is in the test split.
            connected_nodes = edge_subset.get(start)
            label = end in connected_nodes
        except:
            # If not in the test subset assign label to False, otherwise True.
            label=False
        scores.append(value)
        labels.append(label)
    return_list.append([scores,labels])

# Create the processess which are waiting to be executed. 
# Monitor your CPU & RAM usage in a Linux shell by running 'top' command.
starttime = time.time()
processes = [None for i in range(args.n_cores)]

for i in range(args.n_cores):
    processes[i] = mp.Process(target=predict_get_scores, args=(train, chosen_heur, split_arrays[i], return_list, edge_subset_dict))
    processes[i].start()
    
# Join the processess.
print('\n')
print('Joining Processess, Performing Computation')
    
for process in processes:
    process.join()
    
print('\nProcessing done, returning list')
# Return the list of predictions.
y = return_list._getvalue()
print('Computation took {} seconds'.format(time.time() - starttime))

# Convert the array to a numpy array for further numpy routines.
print('Reshaping Predictions')
f = np.array(y, dtype='object')
scores,labels = f.T
scores = np.concatenate(scores, axis=0)
labels = np.concatenate(labels, axis=0)

print('Calculating TPR and FPR')
# Compute False and True positive rate from the list of labels and scores.
fpr, tpr, _ = roc_curve(labels, scores)
# Get AUC of ROC curve.
auc = roc_auc_score(labels, scores)

print('AUC of ROC curve is', auc)

if args.plots == 'YES':
    if args.confusion_mat == 'YES':
        print('Binarising the predictions with threshold')
        # Binarise the predictions with given threshold.
        binarised = np.where(np.array(scores) > args.thresh_bin, 1, 0)
        # Compute the Confusion Matrix, which can take a lot of time
        print('Computing Confusion Matrix')
        cm = confusion_matrix(labels, binarised)

    print('Plotting...')

    # Plot the ROC curve
    plt.figure()
    sns.lineplot(x=fpr, y=tpr, ci=None)
    plt.title("{0} {1} {2} {3}".format('ROC Curve for', chosen_heur.__name__, '- AUC:',str(auc)[:5]))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    if args.confusion_mat == 'YES':
        # Plot the Confusion Matrix
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='0.1f')
        plt.title("{0} {1}".format('Confusion Matrix for', chosen_heur.__name__))
        plt.xlabel('Ground Truth')
        plt.ylabel('Predicted Values')
        plt.show()