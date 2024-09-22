import json
import networkx as nx
from karateclub import Graph2Vec
from findingSlice import Slice
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# Function to load slices from a JSON file
def load_slices(filepath):
    """
    Load slice data from a given JSON file.

    Args:
    filepath (str): Path to the JSON file containing the slice data.

    Returns:
    dict: Loaded JSON data as a dictionary.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)  # Load JSON data from file
    return data  # Return the loaded data

# Function to generate graph embeddings from the slice data
def generate_embeddings(slices_data):
    """
    Generate graph embeddings using the Graph2Vec algorithm.

    Args:
    slices_data (dict): Dictionary containing the slice data.

    Returns:
    np.array: Array of graph embeddings.
    np.array: Array of corresponding labels.
    """
    # Initialize Graph2Vec model with specified parameters
    g2v_model = Graph2Vec(dimensions=128, wl_iterations=4, epochs=5000, learning_rate=0.5)

    graphs = []  # List to hold graph representations
    labels = []  # List to hold corresponding labels
    slice_obj = Slice()  # Initialize Slice object to extract graph structures

    # Loop over the slice data to extract graphs and labels
    for data_index in slices_data:
        label = int(data_index.split(',')[1])  # Extract label from the data
        labels.append(label)  # Add label to the list
        block_slice = slices_data[data_index]  # Get slice data
        G = slice_obj.extract_graph_from_slice(block_slice)  # Extract graph from slice
        graphs.append(G)  # Add graph to the list

    # Fit Graph2Vec model to the list of graphs
    g2v_model.fit(graphs)
    
    # Retrieve the embeddings learned by the Graph2Vec model
    embeddings = g2v_model.get_embedding()
    
    # Return the embeddings and labels as numpy arrays
    return np.array(embeddings), np.array(labels)

# Function to train a RandomForest classifier
def train_random_forest(X, y):
    """
    Train a RandomForest classifier with oversampling (SMOTE) on the graph embeddings.

    Args:
    X (np.array): Graph embeddings (features).
    y (np.array): Labels corresponding to the graph embeddings.
    """
    # Use SMOTE to handle class imbalance by oversampling the minority class
    smote = SMOTE(random_state=0)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split data into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    # Initialize the RandomForest classifier with specific hyperparameters
    model = RandomForestClassifier(n_estimators=150, max_depth=20, class_weight='balanced', random_state=42)

    # Train the RandomForest model on the training data
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Print classification report (precision, recall, F1-score)
    print("Classification Report (Test):")
    print(classification_report(y_test, y_pred))

    # Compute predicted probabilities for ROC-AUC score
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate and print the ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f'ROC AUC: {roc_auc:.2f}')

# Main function to load data, generate embeddings, and train the model
def main():
    # Example path to the synthetic slice data in JSON format
    input_filepath = 'examples/synthetic_slice.json'

    # Load slices data from the JSON file
    slices_data = load_slices(input_filepath)

    # Generate graph embeddings and corresponding labels
    embeddings, labels = generate_embeddings(slices_data)

    # Train the RandomForest model with the generated embeddings and labels
    train_random_forest(embeddings, labels)

# Entry point to run the main function
if __name__ == '__main__':
    main()
