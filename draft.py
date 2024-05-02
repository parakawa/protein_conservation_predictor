from transformers import AutoTokenizer, EsmModel
import torch
from Bio import SeqIO
import os

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

# Función para generar los embeddings vectors de una secuencia


def generate_embedding(sequence):
    inputs = tokenizer(sequence, return_tensors="pt")
    outputs = model(**inputs)
    # Squeeze para eliminar la dimensión de lote
    embedding = outputs.last_hidden_state.squeeze(0)
    # Eliminar la primera y última fila
    embedding = embedding[1:-1, :]
    return embedding


# Crear directorio para almacenar los embeddings

# Leer el archivo fasta y procesar las secuencias
fasta_file = "curated_dataset/sequences.fasta"
count = 0
for record in SeqIO.parse(fasta_file, "fasta"):
    count = count + 1
    sequence_id = record.id.split('/')[0]
    # sequence_id = sequence_id.replace('.', '')
    sequence_id_complement = record.id.split('/')[1]
    sequence = str(record.seq)

    # Generar el embedding para la secuencia actual
    embedding = generate_embedding(sequence)
    output_file = f"curated_dataset/individual_embeddings_original/{sequence_id}.pt"
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Guardar el embedding en un archivo .pt dentro del directorio de la secuencia
    torch.save(embedding, output_file)

    print(
        f"Embedding {count} generado y guardado para la secuencia {sequence_id}")

print("Proceso completado.")
