from src.main import load_slices, generate_embeddings, train_random_forest

# Load synthetic data
slices_data = load_slices('examples/synthetic_slice.json')

# Generate embeddings
X, y = generate_embeddings(slices_data)

# Train Random Forest model
train_random_forest(X, y)
