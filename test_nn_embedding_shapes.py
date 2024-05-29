import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the embedding layer
num_embeddings = 15000  # Size of the vocabulary
embedding_dim = 512    # Dimension of the embedding vectors
embedding_layer = nn.Embedding(num_embeddings, embedding_dim).to(device)

# Example input: a batch of sequences with indices
input_indices = torch.tensor([[1, 2, 3, 4] * 100, [4, 5, 6,7]* 100] * 4).to(torch.long).to(device)

# Apply the embedding layer
output_embeddings = embedding_layer(input_indices)

print("Input indices shape:", input_indices.shape)
print("Output embeddings shape:", output_embeddings.shape)