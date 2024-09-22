import json
import networkx as nx
from karateclub import Graph2Vec
from findingSlice import Slice
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

def load_slices(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def generate_embeddings(slices_data):
    g2v_model = Graph2Vec(dimensions=128, wl_iterations=4, epochs=5000, learning_rate=0.5)
    graphs = []
    labels = []
    slice_obj = Slice()

    for data_index in slices_data:
        label = int(data_index.split(',')[1])
        labels.append(label)
        block_slice = slices_data[data_index]
        G = slice_obj.extract_graph_from_slice(block_slice)
        graphs.append(G)

    g2v_model.fit(graphs)
    embeddings = g2v_model.get_embedding()
    return np.array(embeddings), np.array(labels)

def train_random_forest(X, y):
    smote = SMOTE(random_state=0)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=150, max_depth=20, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Classification Report (Test):")
    print(classification_report(y_test, y_pred))

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f'ROC AUC: {roc_auc:.2f}')

def main():
    # Example synthetic data path
    input_filepath = 'examples/synthetic_slice.json'
    slices_data = load_slices(input_filepath)
    embeddings, labels = generate_embeddings(slices_data)
    train_random_forest(embeddings, labels)

if __name__ == '__main__':
    main()
