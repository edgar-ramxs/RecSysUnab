import pandas as pd

def show_dataframe_dimension(dataframe: pd.DataFrame) -> str:
    """
    Displays the dimensions of the dataframe, including the number of rows, columns, and column names.
    
    Parameters:
    - dataframe (pd.DataFrame): The dataframe for which to display dimensions.

    Returns:
    - str: A formatted string with the number of rows, columns, and column names in the dataframe.
    """
    dimension_info = (
        f"[+] Number of Rows: {dataframe.shape[0]}\n"
        f"[+] Number of Columns: {dataframe.shape[1]}\n"
        f"[+] Column Names: {list(dataframe.columns)}\n"
    )
    return dimension_info


def show_column_data_types(dataframe: pd.DataFrame) -> dict:
    """
    Displays the data types of each column in the dataframe, categorized into numeric and categorical columns.
    
    Parameters:
    - dataframe (pd.DataFrame): The dataframe for which to display column data types.

    Returns:
    - dict: A dictionary with two keys:
        - "Numeric Columns": List of column names with numeric data types (int or float).
        - "Categorical Columns": List of column names with categorical data types (object).
    """
    data_types = {
        "Numeric Columns": dataframe.select_dtypes(include=['int', 'float']).columns.tolist(),
        "Categorical Columns": dataframe.select_dtypes(include=['object']).columns.tolist()
    }
    return data_types


def show_missing_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Displays the amount and percentage of missing data in each column of the dataframe.

    Parameters:
    - dataframe (pd.DataFrame): The dataframe for which to analyze missing data.

    Returns:
    - pd.DataFrame: A new DataFrame with three columns:
        - 'Column': The name of each column with missing data.
        - 'Missing Values': The number of missing values in each column.
        - 'Percentage': The percentage of missing values in each column, relative to the total row count.
    """
    missing_data = dataframe.isna().sum().where(lambda x: x > 0).dropna().reset_index()
    missing_data.columns = ['Column', 'Missing Values']
    missing_data['Percentage'] = (missing_data['Missing Values'] / dataframe.shape[0]) * 100
    return missing_data.sort_values(by='Percentage')


def rename_dataframe_column(column_name: str) -> str:
    """
    Converts a column name to lowercase and replaces spaces or hyphens with underscores.

    Parameters:
    - column_name (str): The original column name to be modified.

    Returns:
    - str: The transformed column name in lowercase with underscores instead of spaces or hyphens.
    """
    return column_name.replace(' ', '_').replace('-', '_').lower()


def show_correlation_columns(dataframe: pd.DataFrame, column_feature: str, top_n_columns: int = 20) -> pd.Series:
    """
    Displays the top N columns most correlated with a specified feature.

    Parameters:
    - dataframe (pd.DataFrame): The dataframe containing the data.
    - column_feature (str): The column name to which correlations should be calculated.
    - top_n_columns (int): The number of top correlated columns to return (default is 20).

    Returns:
    - pd.Series: A sorted Series of the top N correlated columns with their correlation values.
    """
    corr_matrix = dataframe.select_dtypes(include=['int', 'float']).corr()
    return corr_matrix[column_feature].sort_values(ascending=False)[:top_n_columns]


def binary_to_integer(binary_string: str) -> int:
    """
    Converts a binary string to its integer equivalent.

    Parameters:
    - binary_string (str): A string representing a binary number (e.g., '101').

    Returns:
    - int: The integer value of the binary string.
    """
    return int(binary_string, 2)


def integer_to_binary(integer_value: int) -> str:
    """
    Converts an integer to its binary string representation.

    Parameters:
    - integer_value (int): An integer to be converted to binary.

    Returns:
    - str: The binary string representation of the integer (without the '0b' prefix).
    """
    return bin(integer_value)[2:]
