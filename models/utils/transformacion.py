import re

from typing import Dict, List, Optional, Text, Tuple
from pandas import DataFrame, Series, concat
from ast import literal_eval

from surprise import accuracy
from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise.model_selection import cross_validate



def crear_lista_ejercicios(n_ejercicios: int, ejercicios_realizados: List[int]) -> List[int]:
    """Crea una lista binaria que indica si un ejercicio fue realizado o no.

    Parámetros:
    - n_ejercicios (int): Número total de ejercicios.
    - ejercicios_realizados (List[int]): Lista de ejercicios realizados.

    Retorna:
    - List[int]: Lista binaria de tamaño n_ejercicios.
    """
    lista_ejercicios = [0] * n_ejercicios
    for ejercicio in ejercicios_realizados:
        if 0 <= ejercicio < n_ejercicios:
            lista_ejercicios[ejercicio] = 1
    return lista_ejercicios


def crear_matriz_factorizacion(df_usuarios: DataFrame, df_items: DataFrame, columna_usuario: str = 'id_estudiante') -> DataFrame:
    """Genera una matriz de factorización de usuarios por ejercicios en formato binario.

    Parámetros:
    - df_usuarios (DataFrame): DataFrame que contiene la información de los usuarios.
    - df_items (DataFrame): DataFrame que contiene la información de los ejercicios.
    - columna_usuario (str): Nombre de la columna que identifica a los usuarios.

    Retorna:
    - DataFrame: Matriz de factorización de usuarios por ejercicios.
    """
    n_items = len(df_items)
    matriz_factorizacion = DataFrame(columns=[f"e{i}" for i in range(n_items)])

    for indice, fila in df_usuarios.iterrows():
        ejercicios = []
        
        if not fila['ejercicios_hito1'] == -1:
            if isinstance(fila['ejercicios_hito1'], list):
                ejercicios += fila['ejercicios_hito1']
            else:
                ejercicios += literal_eval(fila['ejercicios_hito1'])
            
        if not fila['ejercicios_hito2'] == -1:
            if isinstance(fila['ejercicios_hito2'], list):
                ejercicios += fila['ejercicios_hito2']
            else:
                ejercicios += literal_eval(fila['ejercicios_hito2'])

        if not fila['ejercicios_hito3'] == -1:
            if isinstance(fila['ejercicios_hito3'], list):
                ejercicios += fila['ejercicios_hito3']
            else:
                ejercicios += literal_eval(fila['ejercicios_hito3'])

        if not fila['ejercicios_hito4'] == -1:
            if isinstance(fila['ejercicios_hito4'], list):
                ejercicios += fila['ejercicios_hito4']
            else:
                ejercicios += literal_eval(fila['ejercicios_hito4'])


        vector_ejercicios = crear_lista_ejercicios(n_items, ejercicios)
        matriz_factorizacion.loc[len(matriz_factorizacion)] = vector_ejercicios

    matriz_factorizacion[columna_usuario] = df_usuarios[columna_usuario].values
    matriz_factorizacion = matriz_factorizacion[[columna_usuario] + [f"e{i}" for i in range(n_items)]]
    return matriz_factorizacion


def factorizacion_a_calificaciones(df_ejercicios: DataFrame, matriz_factorizacion: DataFrame, columna_usuario: str = 'id_usuario', prefijo_ejercicio: str = 'e') -> DataFrame:
    """
    Transforma una matriz de factorización en un DataFrame de calificaciones basado en interacciones de usuarios.

    Parámetros:
    - df_ejercicios (pd.DataFrame): DataFrame que contiene detalles de los ejercicios.
    - matriz_factorizacion (pd.DataFrame): Matriz de factorización binaria de interacciones usuario-ejercicio.
    - columna_usuario (str): Nombre de la columna en matriz_factorizacion que representa el ID del usuario.
    - prefijo_ejercicio (str): Prefijo utilizado en las columnas de matriz_factorizacion para representar ejercicios.

    Retorna:
    - pd.DataFrame con interacciones usuario-ejercicio y detalles de los ejercicios.
    """
    filas = []
    for _, fila_matriz in matriz_factorizacion.iterrows():
        id_usuario = fila_matriz[columna_usuario]
        for ejercicio in matriz_factorizacion.columns[1:]:
            if fila_matriz[ejercicio] == 1:
                fila_ejercicio = df_ejercicios.iloc[int(ejercicio.lstrip(prefijo_ejercicio))]
                nueva_fila = {columna_usuario: id_usuario}
                nueva_fila.update(fila_ejercicio.to_dict())
                filas.append(nueva_fila)
    return DataFrame(filas)


def calcular_ratio_interacciones(matriz_factorizacion: DataFrame, df_ejercicios: DataFrame) -> DataFrame:
    """
    Calcula el ratio de interacción para cada ejercicio y lo añade al DataFrame df_ejercicios.
    
    Parámetros:
    - matriz_factorizacion (pd.DataFrame): Matriz de factorización con interacciones usuario-ejercicio.
    - df_ejercicios (pd.DataFrame): DataFrame que contiene la información de los ejercicios.

    Retorna:
    - pd.DataFrame con el df_ejercicios actualizado, incluyendo los ratios de interacción.
    """
    # Calculamos el ratio de interacciones para cada ejercicio
    ratios_interaccion = matriz_factorizacion.iloc[:, 1:].sum().values.tolist()

    # Creamos una copia para evitar el SettingWithCopyWarning
    df_ejercicios_copia = df_ejercicios.copy()
    df_ejercicios_copia['ratio_interaccion'] = [n / len(matriz_factorizacion) for n in ratios_interaccion]
    return df_ejercicios_copia


def calcular_puntuacion_fila(caracteristicas: Dict[str, Tuple[float, float]], valores: Dict[str, float], pesos: Dict[str, float]) -> float:
    """
    Calcula una puntuación normalizada utilizando características, valores y pesos.

    Parámetros:
    - caracteristicas (Dict[str, Tuple[float, float]]): Diccionario de rangos de características en el formato {"nombre": (mínimo, máximo)}.
    - valores (Dict[str, float]): Diccionario de valores de las características para la fila.
    - pesos (Dict[str, float]): Diccionario de pesos para cada característica. Los pesos deben sumar 1.

    Retorna:
    - float: Puntuación calculada, normalizada en el rango [0, 1].
    """
    puntuacion = sum(
        pesos.get(caracteristica, 0) * ((valores[caracteristica] - min_val) / (max_val - min_val) if max_val != min_val else 0)
        for caracteristica, (min_val, max_val) in caracteristicas.items()
        if caracteristica in valores
    )
    return puntuacion


def calcular_puntuacion_dataset(dataframe: DataFrame, caracteristicas: Dict[str, Tuple[float, float]], pesos: Dict[str, float], nueva_columna: str = "Puntuación") -> DataFrame:
    """
    Aplica la función `calcular_puntuacion_fila` a cada fila del DataFrame y agrega una nueva columna.

    Parámetros:
    - dataframe (pd.DataFrame): El DataFrame que contiene los datos.
    - caracteristicas (Dict[str, Tuple[float, float]]): Diccionario de rangos de características en el formato {"nombre": (mínimo, máximo)}.
    - pesos (Dict[str, float]): Diccionario de pesos para cada característica en el formato {"nombre": peso}.
    - nueva_columna (str): Nombre de la nueva columna para almacenar la puntuación calculada. El valor por defecto es "Puntuación".

    Retorna:
    - pd.DataFrame: El DataFrame original con una columna adicional para la puntuación.
    """
    # Verificar que los pesos sumen 1
    if not (0.99 <= sum(pesos.values()) <= 1.01):
        raise ValueError("Los pesos deben sumar 1 para que la puntuación esté en el rango [0, 1].")

    # Aplicar la función de puntuación a cada fila utilizando apply
    dataframe[nueva_columna] = dataframe.apply(
        lambda fila: calcular_puntuacion_fila(caracteristicas, fila.to_dict(), pesos),
        axis=1
    )
    return dataframe


def evaluar_algoritmos(datos, algoritmos=None) -> DataFrame:
    """
    Evalúa múltiples algoritmos utilizando validación cruzada y devuelve un DataFrame de referencia.
    
    Parámetros:
    - datos: Conjunto de datos de Surprise.
    - algoritmos: Lista de instancias de algoritmos de Surprise para evaluar. Si es None, utiliza una lista predeterminada.

    Retorna:
    - pd.DataFrame con los puntajes de referencia y el RMSE de prueba para cada algoritmo.
    """
    if algoritmos is None:
        algoritmos = [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]
    
    referencia = []
    for algoritmo in algoritmos:
        resultados = cross_validate(algoritmo, datos, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        tmp = DataFrame.from_dict(resultados).mean(axis=0)
        nombre_algoritmo = Series([str(algoritmo).split(' ')[0].split('.')[-1]], index=['Algoritmo'])
        tmp = concat([tmp, nombre_algoritmo])
        referencia.append(tmp)
    
    return DataFrame(referencia).set_index('Algoritmo').sort_values(by=['test_rmse', 'test_mae'], ascending=[True, True])



def crear_y_evaluar_modelo_surprise(conjunto_entrenamiento, conjunto_prueba, algoritmo):
    """
    Entrena un modelo utilizando el algoritmo especificado y lo evalúa en los datos de prueba.
    
    Parámetros:
    - conjunto_entrenamiento: Conjunto de entrenamiento de Surprise.
    - conjunto_prueba: Conjunto de prueba de Surprise.
    - algoritmo: Instancia de un algoritmo de Surprise.

    Retorna:
    - Algoritmo entrenado con las métricas de precisión impresas.
    """
    modelo = algoritmo
    modelo.fit(conjunto_entrenamiento)
    predicciones = modelo.test(conjunto_prueba)
    accuracy.rmse(predicciones)
    accuracy.mse(predicciones)
    accuracy.mae(predicciones)
    print()
    return modelo


def crear_train_test_con_queries(dataset: DataFrame, matrix: DataFrame, prueba: DataFrame, queries: list[str] = None, id_column: str = 'id_estudiante'):
    """
    Divide dataset en train y test basado en condiciones definidas por queries, y sincroniza otros DataFrames relacionados.

    Parameters:
        dataset (DataFrame): DataFrame principal que contiene los datos completos a dividir.
        matrix (DataFrame): DataFrame relacionado con el dataset, con una columna común de identificación.
        prueba (DataFrame): Otro DataFrame relacionado que será sincronizado con los conjuntos train y test.
        queries (list[str], opcional): Lista de expresiones lógicas para filtrar el dataset y eliminar filas del conjunto train. Si no se proporciona, se utilizan queries por defecto.
        id_column (str): Nombre de la columna que actúa como identificador único en todos los DataFrames.

    Returns:
        tuple: (train_dataset, test_dataset, train_matrix, test_matrix, train_prueba, test_prueba)
    """

    train_dataset = dataset.copy()
    train_dataset['promedio_solemnes'] = (train_dataset['solemne_1'] + train_dataset['solemne_2'] + train_dataset['solemne_3'] + train_dataset['solemne_4']) / 4

    if queries is None:
        queries = [
            " `ejercicios_hito1` == -1 and `ejercicios_hito2` == -1 and `ejercicios_hito3` == -1 and `ejercicios_hito4` == -1 ", 
            " `solemne_1` < 4.0 and `solemne_2` < 4.0 and `solemne_3` < 4.0 and `solemne_4` < 4.0 ",
            " `promedio_solemnes` < 4.0 "
        ]
    
    for query in queries: 
        remove = train_dataset.query(query)
        train_dataset = train_dataset[~train_dataset[id_column].isin(remove[id_column])]
    
    train_dataset = train_dataset.drop(columns=['promedio_solemnes'])
    test_dataset = dataset[~dataset[id_column].isin(train_dataset[id_column])]

    train_matrix = matrix[matrix[id_column].isin(train_dataset[id_column])]
    test_matrix = matrix[matrix[id_column].isin(test_dataset[id_column])]

    train_prueba = prueba[prueba[id_column].isin(train_dataset[id_column])]
    test_prueba = prueba[prueba[id_column].isin(test_dataset[id_column])]

    return train_dataset, test_dataset, train_matrix, test_matrix, train_prueba, test_prueba


def formatear_texto(texto):
    if isinstance(texto, str):
        texto = texto.strip().lower()  # Eliminar espacios y convertir a minúsculas
        texto = re.sub(r'[^\w\s]', '', texto)  # Eliminar caracteres especiales
        texto = ' '.join(texto.split())  # Eliminar múltiplos espacios
    return texto



def calcular_puntaje_personalizado_prueba(row, cols: list[str]) -> int:
    grupo = row[cols]
    puntaje = 15 * (sum(grupo)/len(cols))
    return int(puntaje)












