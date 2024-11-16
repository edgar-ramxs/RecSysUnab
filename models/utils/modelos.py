import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf

from joblib import dump, load
from typing import Dict, List, Optional, Text, Tuple
from pandas import DataFrame, concat, factorize
from surprise import SVD, Reader, Dataset
from surprise.model_selection import cross_validate




#  ███████╗██╗   ██╗██████╗ ██████╗ ██████╗ ██╗███████╗███████╗
#  ██╔════╝██║   ██║██╔══██╗██╔══██╗██╔══██╗██║██╔════╝██╔════╝
#  ███████╗██║   ██║██████╔╝██████╔╝██████╔╝██║███████╗█████╗  
#  ╚════██║██║   ██║██╔══██╗██╔═══╝ ██╔══██╗██║╚════██║██╔══╝  
#  ███████║╚██████╔╝██║  ██║██║     ██║  ██║██║███████║███████╗
#  ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝

class RecommenderSystemSurprise:
    def __init__(self, algorithm = SVD(), model_path: str = 'RecommenderSystemSurprise.joblib'):
        self.data = None
        self.reader = None
        self.model = algorithm
        self.model_path = model_path
        
    def load_data(self, dataframe: DataFrame, rating_scale: Tuple[float, float]) -> None: 
        self.reader = Reader(rating_scale=rating_scale)
        self.data = Dataset.load_from_df(dataframe, self.reader)

    def train(self) -> None:
        if not self.data:
            raise ValueError("Data not loaded. Use 'load_data' to load a DataFrame first.")
        trainset = self.data.build_full_trainset()
        self.model.fit(trainset)
        print("[+] Model trained successfully.")

    def evaluate(self, cv: int = 5) -> dict:
        if not self.data:
            raise ValueError("Data not loaded. Use 'load_data' to load a DataFrame first.")
        results = cross_validate(self.model, self.data, measures=['RMSE', 'MAE'], cv=cv, verbose=True)
        return results

    def predict(self, user_id: int, item_id: int) -> float:
        return self.model.predict(user_id, item_id).est
    
    def get_recommendations(self, user_id: int, user_column: str = 'user_id', item_column: str = 'item_id', n_recommendations: int = 10) -> List[int]:
        dataframe = self.data.df
        items_interactuados = dataframe[dataframe[user_column] == user_id][item_column].unique()
        todos_los_items = dataframe[item_column].unique()
        pares_items = [(user_id, item, 0) for item in set(todos_los_items) - set(items_interactuados)]
        predicciones = self.model.test(pares_items)
        recomendaciones = sorted(predicciones, key=lambda x: x.est, reverse=True)[:n_recommendations]
        return [int(pred.iid) for pred in recomendaciones]

    def save_model(self) -> None:
        dump(self.model, self.model_path)
        print(f"[+] Model saved to {self.model_path}.")

    def load_model(self) -> None:
        self.model = load(self.model_path)
        print(f"[+] Model loaded from {self.model_path}.")

    def update_and_retrain(self, new_dataframe: DataFrame, rating_scale: Tuple[float, float]) -> str:
        if not self.data:
            raise ValueError("Data not loaded. Use 'load_data' to load an initial DataFrame first.")
        original_dataframe = self.data.df
        updated_dataframe = concat([original_dataframe, new_dataframe]).drop_duplicates()
        self.load_data(updated_dataframe, rating_scale=rating_scale)
        self.train()
        print("[+] Model updated and retrained with new data.")

#  ████████╗ █████╗ ██████╗  ██████╗ ███████╗████████╗    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     
#  ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝ ██╔════╝╚══██╔══╝    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     
#     ██║   ███████║██████╔╝██║  ███╗█████╗     ██║       ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     
#     ██║   ██╔══██║██╔══██╗██║   ██║██╔══╝     ██║       ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     
#     ██║   ██║  ██║██║  ██║╚██████╔╝███████╗   ██║       ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗
#     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝       ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝

class TargetModel:
    def __init__(self, input_shape, learning_rate=0.001, dropout_rate=0.2, layer_units=[64, 32]):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(layer_units[0], activation='relu', input_shape=(input_shape,)))
        self.model.add(tf.keras.layers.Dropout(dropout_rate))
        
        for units in layer_units[1:]:
            self.model.add(tf.keras.layers.Dense(units, activation='relu'))
            self.model.add(tf.keras.layers.Dropout(dropout_rate))
        
        self.model.add(tf.keras.layers.Dense(1, activation='linear'))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    
    def train(self, x_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
        return history
    
    def evaluate(self, x_test, y_test):
        loss = self.model.evaluate(x_test, y_test, verbose=0)
        print(f'Loss en el conjunto de prueba: {loss}')
        return loss
    
    def predict(self, x_test):
        predictions = self.model.predict(x_test)
        return predictions

#  ████████╗██╗    ██╗ ██████╗     ████████╗ ██████╗ ██╗    ██╗███████╗██████╗     ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗         ██╗   ██╗ ██╗
#  ╚══██╔══╝██║    ██║██╔═══██╗    ╚══██╔══╝██╔═══██╗██║    ██║██╔════╝██╔══██╗    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║         ██║   ██║███║
#     ██║   ██║ █╗ ██║██║   ██║       ██║   ██║   ██║██║ █╗ ██║█████╗  ██████╔╝    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║         ██║   ██║╚██║
#     ██║   ██║███╗██║██║   ██║       ██║   ██║   ██║██║███╗██║██╔══╝  ██╔══██╗    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║         ╚██╗ ██╔╝ ██║
#     ██║   ╚███╔███╔╝╚██████╔╝       ██║   ╚██████╔╝╚███╔███╔╝███████╗██║  ██║    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗     ╚████╔╝  ██║
#     ╚═╝    ╚══╝╚══╝  ╚═════╝        ╚═╝    ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝      ╚═══╝   ╚═╝

class TwoTowerModelv1(nn.Module):
    def __init__(self, user_input_size: int, item_input_size: int, embedding_size: int = 64):
        super(TwoTowerModelv1, self).__init__()

        # Torre para los usuarios
        self.user_tower = nn.Sequential(
            nn.Linear(user_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_size)
        )

        # Torre para los ejercicios
        self.item_tower = nn.Sequential(
            nn.Linear(item_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_size)
        )

        # Optimizer y loss function
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def forward(self, user_input, item_input):
        # Obtener los embeddings para usuarios y ejercicios
        user_embedding = self.user_tower(user_input)
        item_embedding = self.item_tower(item_input)
        
        # Calcular la similitud como el producto punto entre los embeddings
        score = torch.sum(user_embedding * item_embedding, dim=1)
        return torch.sigmoid(score)

    def train_model(self, df_users: DataFrame, df_items: DataFrame, epochs: int = 30):
        # Extraer las características de los DataFrames y convertirlas en tensores
        # item_input = torch.tensor(df_items.iloc[:, 1:].values).float()  # Datos de ejercicios
        # user_input = torch.tensor(df_users.iloc[:, 1:].values).float()  # Datos de usuarios

        item_input = torch.tensor(df_items.values).float()  # Usar todas las columnas de ejercicios
        user_input = torch.tensor(df_users.values).float()  # Usar todas las columnas de usuarios

        # Crear pares de usuario e ítem
        num_users = len(df_users)
        num_items = len(df_items)

        user_input_expanded = user_input.unsqueeze(1).expand(-1, num_items, -1).reshape(-1, user_input.size(1))     # Replicamos las características del usuario
        item_input_expanded = item_input.repeat(num_users, 1)                                                       # Repetimos las características de los ítems

        # Entrenar el modelo
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Pasar las entradas a través del modelo
            output = self(user_input_expanded, item_input_expanded)

            # Crear etiquetas aleatorias para los pares
            labels = torch.randint(0, 2, (len(output),)).float()  # Ajustar el tamaño de las etiquetas

            # Calcular la pérdida
            loss = self.criterion(output, labels)

            # Retropropagación
            loss.backward()

            # Actualizar los pesos
            self.optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    def predict(self, df_users: DataFrame, df_items: DataFrame):
        # Extraer las características de los DataFrames y convertirlas en tensores
        # item_input = torch.tensor(df_items.iloc[:, 1:].values).float()  # Datos de ejercicios
        # user_input = torch.tensor(df_users.iloc[:, 1:].values).float()  # Datos de usuarios

        item_input = torch.tensor(df_items.values).float()  # Datos de ejercicios
        user_input = torch.tensor(df_users.values).float()  # Datos de usuarios

        # Crear pares de usuario e ítem
        num_users = len(df_users)
        num_items = len(df_items)

        user_input_expanded = user_input.unsqueeze(1).expand(-1, num_items, -1).reshape(-1, user_input.size(1))     # Replicamos las características del usuario
        item_input_expanded = item_input.repeat(num_users, 1)                                                       # Repetimos las características de los ítems

        # Realizar la predicción
        with torch.no_grad():
            predictions = self(user_input_expanded, item_input_expanded)

        return predictions

#  ████████╗██╗    ██╗ ██████╗     ████████╗ ██████╗ ██╗    ██╗███████╗██████╗     ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗         ██╗   ██╗██████╗ 
#  ╚══██╔══╝██║    ██║██╔═══██╗    ╚══██╔══╝██╔═══██╗██║    ██║██╔════╝██╔══██╗    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║         ██║   ██║╚════██╗
#     ██║   ██║ █╗ ██║██║   ██║       ██║   ██║   ██║██║ █╗ ██║█████╗  ██████╔╝    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║         ██║   ██║ █████╔╝
#     ██║   ██║███╗██║██║   ██║       ██║   ██║   ██║██║███╗██║██╔══╝  ██╔══██╗    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║         ╚██╗ ██╔╝██╔═══╝ 
#     ██║   ╚███╔███╔╝╚██████╔╝       ██║   ╚██████╔╝╚███╔███╔╝███████╗██║  ██║    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗     ╚████╔╝ ███████╗
#     ╚═╝    ╚══╝╚══╝  ╚═════╝        ╚═╝    ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝      ╚═══╝  ╚══════╝


class TwoTowerModelv2(nn.Module):
    def __init__(self, df_users: DataFrame, df_items: DataFrame, embedding_size: int = 64):
        super(TwoTowerModelv2, self).__init__()
        
        self.dataframe_users = df_users
        self.dataframe_items = df_items
        self.embedding_size = embedding_size
        self.user_input_size = len(df_users.columns)
        self.item_input_size = len(df_items.columns)

        self.user_tower = nn.Sequential(
            nn.Linear(self.user_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_size)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(self.item_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_size)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def forward(self, user_input, item_input):
        user_embedding = self.user_tower(user_input)
        item_embedding = self.item_tower(item_input)
        score = torch.sum(user_embedding * item_embedding, dim=1)
        return torch.sigmoid(score)

    def train_model(self, df_users: DataFrame, df_items: DataFrame, epochs: int = 30):
        user_input = torch.tensor(df_users.values).float()
        item_input = torch.tensor(df_items.values).float()
        num_users = len(df_users)
        num_items = len(df_items)
        user_input_expanded = user_input.unsqueeze(1).expand(-1, num_items, -1).reshape(-1, user_input.size(1))
        item_input_expanded = item_input.repeat(num_users, 1)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self(user_input_expanded, item_input_expanded)
            labels = torch.randint(0, 2, (len(output),)).float()
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            print(f"[+] Epoch {epoch+1}/{epochs} => Loss: {loss.item():.4f}")

    def predict(self, df_users: DataFrame, df_items: DataFrame):
        user_input = torch.tensor(df_users.values).float()
        item_input = torch.tensor(df_items.values).float()
        num_users = len(df_users)
        num_items = len(df_items)
        user_input_expanded = user_input.unsqueeze(1).expand(-1, num_items, -1).reshape(-1, user_input.size(1))
        item_input_expanded = item_input.repeat(num_users, 1)
        with torch.no_grad():
            predictions = self(user_input_expanded, item_input_expanded)
        return predictions








#      ██████╗ ███████╗███████╗██████╗        ██╗        ██████╗██████╗  ██████╗ ███████╗███████╗    ███╗   ██╗███████╗████████╗██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗
#      ██╔══██╗██╔════╝██╔════╝██╔══██╗       ██║       ██╔════╝██╔══██╗██╔═══██╗██╔════╝██╔════╝    ████╗  ██║██╔════╝╚══██╔══╝██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝
#      ██║  ██║█████╗  █████╗  ██████╔╝    ████████╗    ██║     ██████╔╝██║   ██║███████╗███████╗    ██╔██╗ ██║█████╗     ██║   ██║ █╗ ██║██║   ██║██████╔╝█████╔╝ 
#      ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝     ██╔═██╔═╝    ██║     ██╔══██╗██║   ██║╚════██║╚════██║    ██║╚██╗██║██╔══╝     ██║   ██║███╗██║██║   ██║██╔══██╗██╔═██╗ 
#      ██████╔╝███████╗███████╗██║         ██████║      ╚██████╗██║  ██║╚██████╔╝███████║███████║    ██║ ╚████║███████╗   ██║   ╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗
#      ╚═════╝ ╚══════╝╚══════╝╚═╝         ╚═════╝       ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝    ╚═╝  ╚═══╝╚══════╝   ╚═╝    ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝

class RecommenderSystemDCNv2:
    def __init__(self, df_interactions, embedding_size=10, learning_rate=0.01):
        # Remapeo de ID de estudiantes e ID de ejercicios a índices consecutivos
        df_interactions['user_index'] = factorize(df_interactions['id_estudiante'])[0]
        df_interactions['item_index'] = factorize(df_interactions['id_ejercicio'])[0]
        
        # Guardar el DataFrame y el número único de usuarios y de ítems
        self.df_interactions = df_interactions
        self.num_users = df_interactions['user_index'].nunique()
        self.num_items = df_interactions['item_index'].nunique()
        
        # Convertir los datos a tensores de PyTorch
        self.user_ids = torch.tensor(df_interactions['user_index'].values).long()
        self.item_ids = torch.tensor(df_interactions['item_index'].values).long()
        self.labels = torch.tensor(df_interactions['score'].values).float()
        
        # Definir y inicializar el modelo de recomendación
        self.model = self.RecommenderNet(self.num_users, self.num_items, embedding_size)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    class RecommenderNet(nn.Module):
        def __init__(self, num_users, num_items, embedding_size):
            super().__init__()
            self.user_embedding = nn.Embedding(num_users, embedding_size)
            self.item_embedding = nn.Embedding(num_items, embedding_size)

        def forward(self, user_ids, item_ids):
            user_vecs = self.user_embedding(user_ids)
            item_vecs = self.item_embedding(item_ids)
            dot_product = (user_vecs * item_vecs).sum(dim=1)
            return dot_product

    def train(self, epochs=100, print_every=10):
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            preds = self.model(self.user_ids, self.item_ids)
            loss = self.loss_fn(preds, self.labels)
            loss.backward()
            self.optimizer.step()
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def recommend_for_user(self, user_id, num_recommendations=5):
        # Obtener los índices de los ejercicios con los que el usuario ya interactuó
        interacted_items = set(self.df_interactions[self.df_interactions['user_index'] == user_id]['item_index'].values)
        
        # Crear un tensor con los índices de los ejercicios que el usuario no ha realizado
        non_interacted_items = torch.tensor([i for i in range(self.num_items) if i not in interacted_items]).long()
        
        # Verificar si quedan ejercicios para recomendar
        if len(non_interacted_items) == 0:
            print(f"No hay nuevos ejercicios para recomendar al usuario {user_id}.")
            return []
        
        # Generar un tensor de usuario para todos los ejercicios no interactuados
        user_tensor = torch.tensor([user_id] * len(non_interacted_items)).long()
        
        # Hacer predicciones para los ejercicios que el usuario no ha realizado
        self.model.eval()
        with torch.no_grad():
            preds = self.model(user_tensor, non_interacted_items).cpu().numpy()
        
        # Ordenar los ejercicios no interactuados por las predicciones más altas y hacer una copia
        sorted_indices = preds.argsort()[-num_recommendations:][::-1].copy()
        recommended_item_ids = non_interacted_items[sorted_indices]
        
        return recommended_item_ids.tolist()

    def save_model(self, file_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, file_path)
        print(f"Modelo guardado en {file_path}")

    def load_model(self, file_path):
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Modelo cargado desde {file_path}")








