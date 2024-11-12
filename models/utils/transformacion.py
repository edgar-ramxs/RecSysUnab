from typing import Dict, List, Optional, Text, Tuple

from pandas import DataFrame, Series, concat

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
        if 0 <= ejercicio < n_ejercicios:   # Validar que el ejercicio esté en el rango permitido
            lista_ejercicios[ejercicio] = 1
    return lista_ejercicios


def crear_matriz_factorizacion(df_usuarios: DataFrame, df_ejercicios: DataFrame, columna_usuario: str, columnas_ejercicios: List[str]) -> DataFrame:
    """Genera una matriz de factorización de usuarios por ejercicios en formato binario.

    Parámetros:
    - df_usuarios (DataFrame): DataFrame que contiene la información de los usuarios.
    - df_ejercicios (DataFrame): DataFrame que contiene la información de los ejercicios.
    - columna_usuario (str): Nombre de la columna que identifica a los usuarios.
    - columnas_ejercicios (List[str]): Lista de nombres de columnas que contienen los ejercicios.

    Retorna:
    - DataFrame: Matriz de factorización de usuarios por ejercicios.
    """
    n_ejercicios = len(df_ejercicios)
    matriz_factorizacion = DataFrame(columns=[f"e{i}" for i in range(n_ejercicios)])

    # Validar que las columnas existan en df_usuarios
    for col in columnas_ejercicios:
        if col not in df_usuarios.columns:
            raise ValueError(f"La columna '{col}' no existe en df_usuarios.")

    for indice in range(len(df_usuarios)):
        ejercicios_realizados = []
        for col in columnas_ejercicios:
            ejercicios = df_usuarios.iloc[indice][col]
            lista_ejercicios = [int(elemento) for elemento in str(ejercicios).split(":")[1:] if elemento.isdigit()]
            ejercicios_realizados.extend(lista_ejercicios)

        vector_ejercicios = crear_lista_ejercicios(n_ejercicios, ejercicios_realizados)
        matriz_factorizacion.loc[len(matriz_factorizacion)] = vector_ejercicios

    matriz_factorizacion[columna_usuario] = df_usuarios[columna_usuario].values
    matriz_factorizacion = matriz_factorizacion[[columna_usuario] + [f"e{i}" for i in range(n_ejercicios)]]
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
        tmp['Puntaje de Referencia'] = 1 / tmp['test_rmse'] if tmp['test_rmse'] != 0 else float('inf')
        referencia.append(tmp)
    
    return DataFrame(referencia).set_index('Algoritmo').sort_values('test_rmse')


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

























