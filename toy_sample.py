import os
import torch
from Bio import SeqIO
import pandas as pd
import random


def reduce_data(fasta_file, csv_file, reduced_amount, reduced_fasta_file, reduced_csv_file):
    # Read the FASTA file and select a random subset of sequences
    fasta_sequences = {}
    selected_identifiers = []
    with open(fasta_file, "r") as fasta:
        records = list(SeqIO.parse(fasta, "fasta"))
        random.shuffle(records)
        selected_records = records[:reduced_amount]
        for record in selected_records:
            fasta_sequences[record.id] = str(record.seq)
            selected_identifiers.append(record.id)

    # Read the CSV file and store conservation scores for matching identifiers
    csv_data = pd.read_csv(csv_file, delimiter='\t', index_col=0)
    reduced_csv_data = pd.DataFrame(
        columns=csv_data.columns)  # Create an empty DataFrame
    for identifier in selected_identifiers:
        # Search for partial matches in the index
        matching_index = csv_data.index[csv_data.index.str.contains(
            identifier)]
        if len(matching_index) > 0:
            reduced_csv_data = pd.concat(
                [reduced_csv_data, csv_data.loc[matching_index]])

    # Write reduced data to new files
    with open(reduced_fasta_file, "w") as fasta_reduced:
        with open(reduced_csv_file, "w") as csv_reduced:
            for identifier, sequence in fasta_sequences.items():
                fasta_reduced.write(f">{identifier}\n{sequence}\n")
            reduced_csv_data.to_csv(csv_reduced, sep='\t')


# Example usage
# reduce_data("curated_dataset/sequences.fasta", "curated_dataset/conservation_scores_formated.csv", 1000,"curated_dataset/reduced_input.fasta", "curated_dataset/reduced_input.csv")

# esm-extract esm2_t6_8M_UR50D curated_dataset/reduced_input.fasta curated_dataset/example_embeddings_esm2_reduced_input --repr_layers 0 5 6 --include mean per_tok

def get_embeddings_vectors(folder_path):

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
    return vectors


vectors = get_embeddings_vectors('sample_data/example_embeddings_esm2')
print("vectors", vectors)
