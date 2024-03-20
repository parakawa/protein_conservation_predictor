import torch
import torch.nn as nn
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ruta a la carpeta que contiene tus carpetas con archivos .pt
embeddings_folder = 'curated_dataset/example_embeddings_esm2_reduced_input'

# Ruta a tu archivo CSV con los scores de conservación
conservation_scores_file = 'ruta_a_tu_archivo_de_scores.csv'

# Obtener la lista de carpetas en la carpeta de embeddings
embedding_subfolders = [os.path.join(embeddings_folder, folder) for folder in os.listdir(
    embeddings_folder) if os.path.isdir(os.path.join(embeddings_folder, folder))]

# Cargar los datos de embeddings del primer archivo .pt de cada carpeta
embeddings_list = []
for folder in embedding_subfolders:
    embedding_files = [file for file in os.listdir(
        folder) if file.endswith('.pt')]
    if embedding_files:
        first_pt_file = os.path.join(folder, embedding_files[0])
        # Suponiendo que 'representations' contiene los embeddings
        embeddings = torch.load(first_pt_file)['representations']
        embeddings_list.append(embeddings)
    print("embeddings_list", embeddings_list)


""" # Concatenar los embeddings en un solo DataFrame
embeddings_df = pd.concat(embeddings_list, ignore_index=True)
print("embeddings_df", embeddings_df)

# Cargar los scores de conservación
conservation_scores = pd.read_csv(conservation_scores_file)

# Unir los datos basados en las secuencias
data = pd.merge(embeddings_df, conservation_scores,
                left_on='sequence_id', right_on='sequence id', how='inner')

# Seleccionar las características (embeddings) y la variable objetivo (scores de conservación)
# Excluir columnas de identificación y scores de conservación
X = data.iloc[:, 1:-3].values
# Tomar solo las columnas de scores de conservación
y = data.iloc[:, -3:].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Normalizar los datos (esto puede ser importante para la regresión lineal)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir a tensores de PyTorch
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# Definir el modelo de regresión lineal en PyTorch


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


# Inicializar el modelo y definir la función de pérdida y el optimizador
input_size = X_train_tensor.shape[1]
output_size = y_train_tensor.shape[1]
model = LinearRegressionModel(input_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Entrenar el modelo
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass y optimización
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Imprimir la pérdida cada 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluar el modelo en el conjunto de prueba
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f'Final Test Loss: {test_loss.item():.4f}')
 """
