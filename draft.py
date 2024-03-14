import os
import torch

# Path to the folder containing the .pt files
folder_path = 'sample_data/example_embeddings_esm2'

# Get the list of files in the folder
files = os.listdir(folder_path)

# Filter the files to only include .pt files
pt_files = [file for file in files if file.endswith('.pt')]

# Initialize a list to store the vectors
vectors = []

# Load the vectors from each .pt file
for file in pt_files:
    file_path = os.path.join(folder_path, file)
    embeddings = torch.load(file_path)["representations"][6]
    float_vector = embeddings.numpy().flatten().astype(float)
    vectors.append(float_vector)
    print("Vector:", float_vector)
