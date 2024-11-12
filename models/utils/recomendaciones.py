import numpy as np
from typing import Dict, List, Optional, Text, Tuple
from pandas import DataFrame


def recomendaciones_top_n_surprise(modelo, dataframe: DataFrame, columna_usuarios: str = 'user_id', columna_items: str = 'item_id', id_usuario: int = 0, n_recomendaciones: int = 10):
    """
    Genera las N mejores recomendaciones para un usuario especificado basado en un modelo entrenado.
    
    Parámetros:
    - dataframe: pd.DataFrame - DataFrame con interacciones usuario-item.
    - modelo: Modelo entrenado de Surprise.
    - columna_usuarios: str - Nombre de la columna para los IDs de los usuarios en dataframe.
    - columna_items: str - Nombre de la columna para los IDs de los ítems en dataframe.
    - id_usuario: int - ID del usuario para el cual se generan las recomendaciones.
    - n_recomendaciones: int - Número de mejores recomendaciones a obtener.

    Retorna:
    - Lista de IDs de los ítems recomendados para el usuario.
    """
    items_interactuados = dataframe[dataframe[columna_usuarios] == id_usuario][columna_items].unique()
    todos_los_items = dataframe[columna_items].unique()
    pares_items = [(id_usuario, item, 0) for item in set(todos_los_items) - set(items_interactuados)]
    predicciones = modelo.test(pares_items)
    recomendaciones_top_n = sorted(predicciones, key=lambda x: x.est, reverse=True)[:n_recomendaciones]
    return [int(pred.iid) for pred in recomendaciones_top_n]


def recomendaciones_TwoTowerModelv1(prediciones, id_usuario: int, df_ejercicios: DataFrame, matriz_factorizacion: DataFrame, threshold: float = 0.01) -> DataFrame:
    """Recomienda ejercicios para un usuario específico basado en las predicciones, excluyendo los ejercicios con los que ya ha interactuado.

    Parámetros:
    - prediciones: Tensor de predicciones del modelo.
    - id_usuario: ID del usuario para el cual se quieren recomendaciones.
    - df_ejercicios: DataFrame que contiene los ejercicios.
    - matriz_factorizacion: DataFrame que contiene las interacciones en formato binario.
    - threshold: Umbral para la recomendación (por defecto 0.1).

    Retorna:
    - DataFrame con ejercicios recomendados para el usuario, ordenados por predicción de mayor a menor.
    """

    if id_usuario not in matriz_factorizacion['id_estudiante'].values:
        print(f"Error: El usuario {id_usuario} no tiene interacciones previas. No se pueden hacer recomendaciones.")
        return DataFrame()

    prediciones_numpy = prediciones.numpy()
    num_usuarios = len(prediciones_numpy) // len(df_ejercicios)
    
    if id_usuario < 0 or id_usuario >= num_usuarios:
        raise ValueError("id_usuario fuera de rango")
    
    prediciones_usuario = prediciones_numpy[id_usuario::num_usuarios]

    recomendaciones_indices = np.where(prediciones_usuario > threshold)[0]
    recomendaciones_puntaje = prediciones_usuario[recomendaciones_indices]

    recomendaciones_ejercicios = df_ejercicios.iloc[recomendaciones_indices].copy()
    recomendaciones_ejercicios['prediccion'] = recomendaciones_puntaje

    ejercicios_interactuados = matriz_factorizacion[matriz_factorizacion['id_estudiante'] == id_usuario].iloc[:, 1:]
    ejercicios_interactuados_indices = ejercicios_interactuados.columns[ejercicios_interactuados.values[0] == 1].tolist()
    ejercicios_interactuados_indices = [idx[1:] if idx.startswith('e') else idx for idx in ejercicios_interactuados_indices]

    recomendaciones_ejercicios['id_ejercicio'] = recomendaciones_ejercicios['id_ejercicio'].astype(str)
    recomendaciones_ejercicios = recomendaciones_ejercicios[~recomendaciones_ejercicios['id_ejercicio'].isin(ejercicios_interactuados_indices)]
    recomendaciones_ejercicios = recomendaciones_ejercicios.sort_values(by='prediccion', ascending=False).reset_index(drop=True)

    return recomendaciones_ejercicios








