import subprocess
import json
import networkx as nx
from statistics import mean
#import matplotlib.pyplot as plt
from karateclub import Graph2Vec
from findingSlice import Slice
from sklearn.model_selection import cross_validate
#from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import pickle
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt




# Open and load the JSON data

f = open("C:\\Users\\hatwi\\Documents\\Thesis development environment\\Slice files\\Backwardslicing.json")
data = json.load(f)
f.close()
g2v_model = Graph2Vec ( dimensions =128 , wl_iterations =4 , epochs =5000 ,
learning_rate =0.5)
graphs =[]

labels_final = [] 
mainBlocks = []
graphsAndLabels = []


for data_index in data: 
    label = data_index.split(',')[1]
    labels_final.append(label)

    block_slice = data[data_index]
    slice_obj = Slice()
    
    G = nx.DiGraph()
    

    # Create a mapping from SID to index
    sid_to_index = {}
    index_counter = 0

    # Create a set of all SIDs in the data
    all_sids = set(block['SID'] for block in block_slice)

    # Define the list of attributes to check and associate
    attributes = ['OutputSignals', 'OutDataTypeStr', 'OutputDataType', 'UpperLimit', 
                'LowerLimit', 'OutMin', 'OutMax', 'Function', 'IntegrationMethod', 'SampleTime',
                'LimitOutput', 'UpperSaturationLimit', 'LowerSaturationLimit', 'ICPrevOutput', 'ICPrevScaledInput',
                'Operator']
    mainBlock = block_slice [-1]
    # Function to add a node with its attributes
    def add_node_with_attributes(sid, block):
        node_index = sid_to_index[sid]
        
        # Initialize the node attributes with SID
        node_attributes = {'SID': sid}
        node_attributes['BlockType']=block['BlockType']
        # Check and add each attribute from the PropertyDict
        if 'PropertyDict' in block:
            for attr in attributes:
                if attr in block['PropertyDict']:
                    node_attributes[attr] = block['PropertyDict'][attr]
                else:
                    node_attributes[attr] = ''
        
        if block['SID']==mainBlock['SID']:
            node_attributes['is_main'] = True
            #node_attributes['BlockType']=mainBlock['BlockType']
            node_attributes['SliceSize']= len(block_slice)
        

        # Add the node with its attributes
        G.add_node(node_index, **node_attributes)

    # Iterate over each block in the data
    for block in block_slice:
        destinationIDs = block['DestinationBlockSIDs']
        sourceIDs = block['SourceBlockSIDs']
        
        # Map the SID to an index if not already mapped
        if block['SID'] not in sid_to_index:
            sid_to_index[block['SID']] = index_counter
            index_counter += 1
        
        # Add the node for the current block with its attributes
        add_node_with_attributes(block['SID'], block)
        
        # Iterate over destinationIDs to add nodes and edges
        for destID in destinationIDs:
            if destID in all_sids:
                if destID not in sid_to_index:
                    sid_to_index[destID] = index_counter
                    index_counter += 1
                
                # Find the block corresponding to destID to get its properties
                dest_block = next((b for b in block_slice if b['SID'] == destID), None)
                if dest_block:
                    add_node_with_attributes(destID, dest_block)
                    G.add_edge(sid_to_index[block['SID']], sid_to_index[destID])
        
        # Iterate over sourceIDs to add nodes and edges
        for sourceID in sourceIDs:
            if sourceID in all_sids:
                if sourceID not in sid_to_index:
                    sid_to_index[sourceID] = index_counter
                    index_counter += 1
                
                # Find the block corresponding to sourceID to get its properties
                source_block = next((b for b in block_slice if b['SID'] == sourceID), None)
                if source_block:
                    add_node_with_attributes(sourceID, source_block)
                    G.add_edge(sid_to_index[sourceID], sid_to_index[block['SID']])
    
    
    
    graphs.append(G)
    graphsAndLabels.append((G, labels_final))
    
    #mainBlocks.append(mainBlock)
    


g2v_model.fit(graphs)
embedding_final = g2v_model.get_embedding()







X = np.array(embedding)
int_labels = [int(item) for item in labels]
y = np.array(int_labels)

smote = SMOTE(random_state=0)



# Perform stratified k-fold cross-validation
for train_index, test_index in skf.split(X, y):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Apply SMOTE to the training set only
    x_train, y_train = smote.fit_resample(x_train, y_train)


    clf = RandomForestClassifier(max_depth=20, 
                                       min_samples_leaf=1, 
                                       min_samples_split=2, 
                                       n_estimators=150,
                                       random_state=42, 
                                       class_weight='balanced')
    
    clf.fit(x_train, y_train)
    y_pred = clf . predict ( x_test )

    accuracy = accuracy_score(y_test, y_pred)
    #print(f'Accuracy: {accuracy:.2f}')

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[1, 2]).ravel()
    y_pred_proba = clf.predict_proba(x_test)[:, 1]

    precision = precision_score(y_test, y_pred, pos_label=2, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=2, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=2, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba, pos_label=2)
    pr_auc = auc(recall_vals, precision_vals)

    y_train_pred = clf.predict(x_train) 
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    roc_aucs.append(roc_auc)
    pr_aucs.append(pr_auc)
    




# Print average metrics across all folds
print(f'Average Precision: {np.mean(precisions):.2f} ± {np.std(precisions):.2f}')
print(f'Average Recall: {np.mean(recalls):.2f} ± {np.std(recalls):.2f}')
print(f'Average F1 Score: {np.mean(f1s):.2f} ± {np.std(f1s):.2f}')
print(f'Average ROC AUC: {np.mean(roc_aucs):.2f} ± {np.std(roc_aucs):.2f}')
print(f'Average PR AUC: {np.mean(pr_aucs):.2f} ± {np.std(pr_aucs):.2f}')


x_train , x_test , y_train , y_test = train_test_split (X , y
 ,
test_size =0.3)
RF = RandomForestClassifier(max_depth=20, 
                                       min_samples_leaf=1, 
                                       min_samples_split=2, 
                                       n_estimators=150,
                                       random_state=42, 
                                       class_weight='balanced')

x_train, y_train = smote.fit_resample(x_train, y_train)
RF.fit(x_train, y_train)
y_train_pred = RF.predict(x_train)
y_test_pred = RF.predict(x_test)



precision = precision_score(y_test, y_test_pred, pos_label=2, zero_division=0)
recall = recall_score(y_test, y_test_pred, pos_label=2, zero_division=0)
f1 = f1_score(y_test, y_test_pred, pos_label=2, zero_division=0)
y_pred_proba = RF.predict_proba(x_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

balanced_accuracy = (recall_score(y_test, y_test_pred, pos_label=1, zero_division=0) + recall_score(y_test, y_test_pred, pos_label=2, zero_division=0)) / 2



print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'ROC AUC: {roc_auc:.2f}')
print(f'Balanced Accuracy: {balanced_accuracy:.2f}')

print("Classification Report (Train):")
print(classification_report(y_train, y_train_pred))

print("Classification Report (Test):")
print(classification_report(y_test, y_test_pred))

