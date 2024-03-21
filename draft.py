import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


def get_embeddings_vectors_curated_data(folder_path):
    # Initialize a list to store the vectors
    vectors = []
    embeddings = {}

    # Traverse through each folder in the specified directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is a .pt file
            if file.endswith('.pt'):
                file_path = os.path.join(root, file)
                embedding = torch.load(file_path)
                embeddings[embedding["label"]
                           ] = embedding["representations"][6]
                # float_vector = embeddings.numpy().astype(float)
                # vectors.append(float_vector)

    return vectors, embeddings


vectors, embeddings_dict = get_embeddings_vectors_curated_data(
    'curated_dataset/example_embeddings_esm2_reduced_input')
# print("Vectors:", len(vectors[0]))
print("embeddings_dict len", (embeddings_dict["A0A1X7AIY7.1/282-340"]).shape)
print("embeddings_dict", embeddings_dict["A0A1X7AIY7.1/282-340"])
len("STPIRIFANGRRRVEVLRDNRLIYATSVNAGSQEIDTSSFPQGSYQLTIRIFNGSTLEQ")

# Función para cargar los datos desde el CSV y convertirlos en tensores


def load_data(csv_file):
    # Cargar el CSV
    df = pd.read_csv(csv_file, delimiter=',', names=[
                     'sequence id', 'conservation score'], header=0)

    # Reemplazar NaN con 0
    df.fillna(0, inplace=True)

    sequences = df['sequence id'].values
    conservation_scores = df['conservation score'].apply(lambda x: np.array(
        [float(i) for i in x.split() if i != 'nan'], dtype=np.float32)).values

    # Convertir a tensores de PyTorch
    conservation_scores = [torch.tensor(score)
                           for score in conservation_scores]

    return sequences, conservation_scores


sequences, conservation_scores_tensors = load_data(
    'curated_dataset/reduced_input.csv')

# Función para obtener los embeddings correspondientes a las secuencias


def get_embeddings(sequences, embeddings_dict):
    embeddings = []
    for sequence_id in sequences:
        embedding = embeddings_dict[sequence_id]
        print(embedding)
        embeddings.append(embedding)
    embeddings = torch.stack(embeddings)
    return embeddings


def get_embedding(sequence_id, embeddings_dict):
    return embeddings_dict[sequence_id]


# Función para entrenar el modelo utilizando SGD de forma estocástica
def train_model_stochastic(model, optimizer, loss_fn, sequences, conservation_scores):
    model.train()
    for i in range(len(sequences)):
        sequence_id = sequences[i]
        embedding = get_embedding(sequence_id, embeddings_dict)
        embedding_tensor = torch.tensor(
            embedding, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(conservation_scores[i], dtype=torch.float32)

        optimizer.zero_grad()
        output = model(embedding_tensor)

        # Ajustar la longitud del tensor output
        if output.size(1) > label.size(0):
            # Si el tensor output es más largo que label, truncarlo
            output = output[:, :label.size(0)]
        elif output.size(1) < label.size(0):
            # Si el tensor output es más corto que label, rellenarlo con ceros
            output = torch.cat([output, torch.zeros(
                1, label.size(0) - output.size(1))], dim=1)

        loss = loss_fn(output.squeeze(), label)
        loss.backward()
        optimizer.step()


# Definir el modelo de regresión lineal


class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Función para entrenar un modelo durante una época


sequences, conservation_scores = load_data('curated_dataset/reduced_input.csv')


# Evaluación del modelo en el conjunto de validación


def evaluate_model(model, loss_fn, data_loader):
    running_loss = 0.

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)

            running_loss += loss.item()

    return running_loss / len(data_loader)


# Configuración de hiperparámetros
learning_rate = 0.001
num_epochs = 5
batch_size = 32


# Crear el conjunto de datos
dataset = [(embeddings_dict[sequence], conservation_scores) for sequence,
           conservation_scores in zip(sequences, conservation_scores_tensors)]
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size])

# Crear los data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


# Inicializar el modelo, la función de pérdida y el optimizador
model = LinearRegression(input_size=320)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Entrenamiento del modelo
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}:')
    model.train()
    train_model_stochastic(model, optimizer, loss_fn,
                           sequences, conservation_scores)

    # Validación del modelo
    model.eval()
    val_loss = evaluate_model(model, loss_fn, val_loader)

    print(f'Validation Loss: {val_loss}')
