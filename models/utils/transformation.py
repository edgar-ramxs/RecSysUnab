import pandas as pd
from typing import Dict, List, Optional, Text, Tuple
from surprise import accuracy, SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise.model_selection import cross_validate


def create_exercise_list(n_items: int, items: List[int]) -> List[int]:
    """Crea una lista binaria que indica si un ejercicio fue realizado o no.

    Args:
        n_items (int): Número total de ejercicios.
        items (List[int]): Lista de ejercicios realizados.

    Returns:
        List[int]: Lista binaria de tamaño n_items.
    """
    item_list = [0] * n_items
    for item in items:
        if (
            0 <= item < n_items
        ):  # Validar que el ejercicio esté en el rango permitido
            item_list[item] = 1
    return item_list


def create_factorization_matrix(df_users: pd.DataFrame,df_items: pd.DataFrame,user_col: str,items_col: List[str]) -> pd.DataFrame:
    """Genera una matriz de factorización de usuarios por ejercicios en formato binario.

    Args:
        df_users (pd.DataFrame): DataFrame que contiene la información de los usuarios.
        df_items (pd.DataFrame): DataFrame que contiene la información de los ejercicios.
        user_col (str): Nombre de la columna que identifica a los usuarios.
        items_col (List[str]): Lista de nombres de columnas que contienen los ejercicios.

    Returns:
        pd.DataFrame: Matriz de factorización de usuarios por ejercicios.
    """
    n_items = len(df_items)
    matrix_factorization = pd.DataFrame(columns=[f"e{i}" for i in range(n_items)])

    # Validar que las columnas existan en df_users
    for col in items_col:
        if col not in df_users.columns:
            raise ValueError(f"La columna '{col}' no existe en df_users.")

    for index in range(len(df_users)):
        items_performed = []
        for col in items_col:
            items = df_users.iloc[index][col]
            items_list = [
                int(element)
                for element in str(items).split(":")[1:]
                if element.isdigit()
            ]
            items_performed.extend(items_list)

        items_vector = create_exercise_list(n_items, items_performed)
        matrix_factorization.loc[len(matrix_factorization)] = items_vector

    matrix_factorization[user_col] = df_users[user_col].values
    matrix_factorization = matrix_factorization[
        [user_col] + [f"e{i}" for i in range(n_items)]
    ]

    return matrix_factorization


def factorization_to_ratings(df_items: pd.DataFrame, matrix_factorization: pd.DataFrame, user_col: str = 'user_id', item_prefix: str = 'e') -> pd.DataFrame:
    """
    Transforms a factorization matrix into a ratings DataFrame based on user interactions.
    
    Parameters:
    - df_items: pd.DataFrame - DataFrame containing item details.
    - matrix_factorization: pd.DataFrame - Binary factorization matrix of user-item interactions.
    - user_col: str - Column name in matrix_factorization that represents user ID.
    - item_prefix: str - Prefix used in matrix_factorization columns to represent items.

    Returns:
    - pd.DataFrame with user-item interactions and item details.
    """
    rows = []
    for _, row_matrix in matrix_factorization.iterrows():
        user_id = row_matrix[user_col]
        for item in matrix_factorization.columns[1:]:
            if row_matrix[item] == 1:
                row_item = df_items.iloc[int(item.lstrip(item_prefix))]
                new_row = {user_col: user_id}
                new_row.update(row_item.to_dict())
                rows.append(new_row)
    return pd.DataFrame(rows)


def calculate_ratio_of_interactions(matrix_factorization: pd.DataFrame, df_items: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates interaction ratio for each item and adds it to the df_items DataFrame.
    
    Parameters:
    - matrix_factorization: pd.DataFrame - Factorization matrix with user-item interactions.
    - df_items: pd.DataFrame - DataFrame containing item information.

    Returns:
    - pd.DataFrame with updated df_items including interaction ratios.
    """
    interaction_ratios = matrix_factorization.sum().values.tolist()[1:]
    df_items['interaction_ratio'] = [n / len(matrix_factorization) for n in interaction_ratios]
    return df_items


def create_and_evaluate_model(trainset, testset, algorithm):
    """
    Trains a model using the specified algorithm and evaluates it on test data.
    
    Parameters:
    - trainset: Surprise training set.
    - testset: Surprise test set.
    - algorithm: Algorithm instance from Surprise.

    Returns:
    - Trained algorithm with accuracy metrics printed.
    """
    algo = algorithm
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions)
    accuracy.mse(predictions)
    accuracy.mae(predictions)
    print()
    return algo


def evaluate_algorithms(data, algorithms=None) -> pd.DataFrame:
    """
    Evaluates multiple algorithms using cross-validation and returns a benchmark DataFrame.
    
    Parameters:
    - data: Surprise dataset.
    - algorithms: List of Surprise algorithm instances to evaluate. If None, uses a default list.

    Returns:
    - pd.DataFrame with benchmark scores and test RMSE for each algorithm.
    """
    if algorithms is None:
        algorithms = [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]
    benchmark = []
    for algorithm in algorithms:
        results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        algorithm_name = pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm'])
        tmp = pd.concat([tmp, algorithm_name])
        tmp['Benchmark Score'] = 1 / tmp['test_rmse'] if tmp['test_rmse'] != 0 else float('inf')
        benchmark.append(tmp)
    return pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')


def get_top_n_recommendations(model, df: pd.DataFrame, feature_users: str = 'user_id', feature_items: str = 'item_id', user_id: int = 0, n_recommenders: int = 10):
    """
    Generates top N recommendations for a specified user based on a trained model.
    
    Parameters:
    - df: pd.DataFrame - DataFrame with user-item interactions.
    - model: Trained Surprise model.
    - feature_users: str - Column name for user IDs in df.
    - feature_items: str - Column name for item IDs in df.
    - user_id: int - User ID for whom recommendations are generated.
    - n_recommenders: int - Number of top recommendations to retrieve.

    Returns:
    - List of recommended item IDs for the user.
    """
    items_interacted = df[df[feature_users] == user_id][feature_items].unique()
    all_items = df[feature_items].unique()
    items_pairs = [(user_id, item, 0) for item in set(all_items) - set(items_interacted)]
    predictions = model.test(items_pairs)
    top_n_recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:n_recommenders]
    return [str(pred.iid) for pred in top_n_recommendations]


def calculate_score_row(features: Dict[str, Tuple[float, float]], values: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Calculates a normalized score using features, values, and weights.

    Parameters:
    - features (Dict[str, Tuple[float, float]]): Dictionary of feature ranges in the format {"name": (min, max)}.
    - values (Dict[str, float]): Dictionary of feature values for the row.
    - weights (Dict[str, float]): Dictionary of weights for each feature. Weights should sum to 1.

    Returns:
    - float: Calculated score, normalized in the range [0, 1].
    """
    score = 0
    for feature, (min_val, max_val) in features.items():
        if feature in values:
            # Normalize the feature value in the range [0, 1]
            normalized_value = (values[feature] - min_val) / (max_val - min_val) if max_val != min_val else 0
            # Add the weighted contribution to the score
            score += weights.get(feature, 0) * normalized_value
    return score

def calculate_score_dataset(df: pd.DataFrame, features: Dict[str, Tuple[float, float]], weights: Dict[str, float], new_column: str = "Score") -> pd.DataFrame:
    """
    Applies the `calculate_score_row` function to each row in the DataFrame and adds a new column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - features (Dict[str, Tuple[float, float]]): Dictionary of feature ranges in the format {"name": (min, max)}.
    - weights (Dict[str, float]): Dictionary of weights for each feature in the format {"name": weight}.
    - new_column (str): Name of the new column to store the calculated score. Default is "Score".

    Returns:
    - pd.DataFrame: The original DataFrame with an additional column for the score.
    """
    # Check that weights sum to 1
    if not (0.99 <= sum(weights.values()) <= 1.01):
        raise ValueError("Weights must sum to 1 for the score to be in the range [0, 1].")

    # Apply the score function to each row
    df[new_column] = df.apply(lambda row: calculate_score_row(features, row.to_dict(), weights), axis=1)
    return df

































