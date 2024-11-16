from pandas import DataFrame, Series
# import numpy as np 
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt


def dimensiones_dataframe(dataframe: DataFrame) -> str:
    """
    Muestra las dimensiones del dataframe, incluyendo el número de filas, columnas y nombres de las columnas.

    Parámetros:
    - dataframe (DataFrame): El dataframe para el cual se desean mostrar las dimensiones.

    Retorna:
    - str: Una cadena formateada con el número de filas, columnas y nombres de las columnas en el dataframe.
    """
    informacion_dimension = (
        f"[+] Número de Filas: {dataframe.shape[0]}\n"
        f"[+] Número de Columnas: {dataframe.shape[1]}\n"
        f"[+] Nombres de Columnas: {list(dataframe.columns)}\n"
    )
    return informacion_dimension


def tipos_datos_dataframe(dataframe: DataFrame) -> dict:
    """
    Muestra los tipos de datos de cada columna en el dataframe, categorizados en columnas numéricas y categóricas.
    
    Parámetros:
    - dataframe (pd.DataFrame): El dataframe para el cual se desean mostrar los tipos de datos de las columnas.

    Retorna:
    - dict: Un diccionario con dos claves:
        - "Columnas Numéricas": Lista de nombres de columnas con tipos de datos numéricos (int o float).
        - "Columnas Categóricas": Lista de nombres de columnas con tipos de datos categóricos (object).
    """
    tipos_datos = {
        "Columnas Numéricas": dataframe.select_dtypes(include=['int', 'float']).columns.tolist(),
        "Columnas Categóricas": dataframe.select_dtypes(include=['object']).columns.tolist()
    }
    return tipos_datos


def datos_faltantes_dataframe(dataframe: DataFrame) -> DataFrame:
    """
    Muestra la cantidad y el porcentaje de datos faltantes en cada columna del dataframe.
    
    Parámetros:
    - dataframe (pd.DataFrame): El dataframe para analizar los datos faltantes.

    Retorna:
    - pd.DataFrame: Un nuevo DataFrame con tres columnas:
        - 'Columna': El nombre de cada columna con datos faltantes.
        - 'Valores Faltantes': La cantidad de valores faltantes en cada columna.
        - 'Porcentaje': El porcentaje de valores faltantes en cada columna, en relación al total de filas.
    """
    datos_faltantes = dataframe.isna().sum().where(lambda x: x > 0).dropna().reset_index()
    datos_faltantes.columns = ['Columna', 'Valores Faltantes']
    datos_faltantes['Porcentaje'] = (datos_faltantes['Valores Faltantes'] / dataframe.shape[0]) * 100
    return datos_faltantes.sort_values(by='Porcentaje')


def renombrar_columnas_dataframe(nombre_columna: str) -> str:
    """
    Convierte el nombre de una columna a minúsculas y reemplaza espacios o guiones por guiones bajos.
    
    Parámetros:
    - nombre_columna (str): El nombre original de la columna a modificar.

    Retorna:
    - str: El nombre de la columna transformado en minúsculas con guiones bajos en lugar de espacios o guiones.
    """
    return nombre_columna.replace(' ', '_').replace('-', '_').lower()


def correlacion_variables_dataframe(dataframe: DataFrame, columna_objetivo: str, top_n_columnas: int = 20) -> Series:
    """
    Muestra las N columnas más correlacionadas con una característica específica.

    Parámetros:
    - dataframe (DataFrame): El dataframe que contiene los datos.
    - columna_objetivo (str): El nombre de la columna para calcular las correlaciones.
    - top_n_columnas (int): La cantidad de columnas con mayor correlación a retornar (por defecto es 20).

    Retorna:
    - Series: Una Serie ordenada de las N columnas más correlacionadas con sus valores de correlación.
    """
    matriz_corr = dataframe.select_dtypes(include=['int', 'float']).corr()
    return matriz_corr[columna_objetivo].sort_values(ascending=False)[:top_n_columnas]


def binario_a_entero(cadena_binaria: str) -> int:
    """
    Convierte una cadena binaria a su equivalente entero.

    Parámetros:
    - cadena_binaria (str): Una cadena que representa un número binario (por ejemplo, '101').

    Retorna:
    - int: El valor entero de la cadena binaria.
    """
    return int(cadena_binaria, 2)


def entero_a_binario(valor_entero: int) -> str:
    """
    Convierte un entero a su representación en cadena binaria.

    Parámetros:
    - valor_entero (int): Un entero a convertir en binario.

    Retorna:
    - str: La representación en cadena binaria del entero (sin el prefijo '0b').
    """
    return bin(valor_entero)[2:]





