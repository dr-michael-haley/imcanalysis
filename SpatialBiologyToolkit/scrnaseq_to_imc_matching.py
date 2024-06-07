import anndata as ad
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.inspection import permutation_importance
from pathlib import Path
from typing import Union, Tuple, Dict, List

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    scale: str = 'ss',
    adata: ad.AnnData = None,
    obs: str = None,
    neurons: int = 16,
    optimizer: str = 'adam',
    batch_size: int = 32
) -> Union[Sequential, Tuple[Sequential, Dict[int, str]]]:
    """
    Train a neural network model to classify cell populations.

    Parameters
    ----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target labels.
    scale : str
        Scaling method ('ss' for StandardScaler, 'mm' for MinMaxScaler).
    adata : ad.AnnData, optional
        Annotated data matrix.
    obs : str, optional
        Observation column in AnnData.
    neurons : int
        Number of neurons in hidden layers.
    optimizer : str
        Optimizer for training the model.
    batch_size : int
        Batch size for training.

    Returns
    -------
    Union[Sequential, Tuple[Sequential, Dict[int, str]]]
        Trained model and optionally a mapping of codes to categories.
    """
    if scale == 'ss':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scale == 'mm':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    number_features = X.shape[1]
    number_groups = y.max() + 1

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Neural Network Model
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(number_features,)),  # Input layer
        Dense(neurons, activation='relu'),  # Hidden layer
        Dense(number_groups, activation='softmax')  # Output layer
    ])

    # Compile the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=batch_size)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy}")

    if adata is None:
        return model
    else:
        # Also return a dictionary of how the codes map to original populations
        codes = adata.obs[obs].cat.codes
        codes_to_category = {code: category for code, category in zip(codes, adata.obs[obs])}
        return model, codes_to_category

def predict_groups(
    model: Sequential,
    X: np.ndarray,
    codes_to_category: Dict[int, str] = None,
    scale: str = 'ss',
    confidence_threshold: float = None,
    below_limit: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict cell populations using the trained model.

    Parameters
    ----------
    model : Sequential
        Trained neural network model.
    X : np.ndarray
        Input features for prediction.
    codes_to_category : Dict[int, str], optional
        Mapping of codes to categories.
    scale : str
        Scaling method ('ss' for StandardScaler, 'mm' for MinMaxScaler).
    confidence_threshold : float, optional
        Threshold for prediction confidence.
    below_limit : int, optional
        Value to assign if confidence is below threshold.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Predicted classes and their confidence scores.
    """
    if scale == 'ss':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scale == 'mm':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    # Predict classes for the new data
    predictions = model.predict(X)

    if confidence_threshold:
        predicted_classes = []
        for pred in predictions:
            max_probability = np.max(pred)
            predicted_class = np.argmax(pred) if max_probability >= confidence_threshold else below_limit
            predicted_classes.append(predicted_class)
        predicted_classes = np.array(predicted_classes)
    else:
        predicted_classes = np.argmax(predictions, axis=1)
        prediction_confidence = np.max(predictions, axis=1)
        
    if codes_to_category:
        predicted_classes = pd.Series(predicted_classes).map(codes_to_category)
    
    return predicted_classes, prediction_confidence

def evaluate_model(model: Sequential, X: np.ndarray, y: np.ndarray) -> float:
    """
    Evaluate the accuracy of the model.

    Parameters
    ----------
    model : Sequential
        Trained neural network model.
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target labels.

    Returns
    -------
    float
        Accuracy of the model.
    """
    loss, accuracy = model.evaluate(X, y, verbose=0)
    return accuracy

def rank_feature_importance(model: Sequential, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Rank feature importance using permutation importance.

    Parameters
    ----------
    model : Sequential
        Trained neural network model.
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target labels.
    feature_names : List[str]
        List of feature names.

    Returns
    -------
    pd.DataFrame
        DataFrame containing feature names and their importance scores.
    """
    results = permutation_importance(model, X, y, scoring=evaluate_model)
    importance_scores = results.importances_mean
    return pd.DataFrame(zip(feature_names, importance_scores), columns=['Feature', 'Mean Importance Score'])

def data_mapping(
    adata_sc: ad.AnnData, 
    adata_imc: ad.AnnData, 
    gene_protein_mapping: pd.DataFrame,
    gene_protein_mapping_usecol: str,
    gene_protein_mapping_sc_col: str = 'snRNAseq',
    gene_protein_mapping_imc_col: str = 'IMC',
    sc_pop_exclude: List[str] = [],
    sc_obs: str = 'population',
    imc_pop_exclude: List[str] = [],
    imc_obs: str = 'population',
    scale: str = 'ss',
    confidence_threshold: float = None,
    neurons: int = 64, 
    optimizer: str = 'adam',
    batch_size: int = 32
) -> None:
    """
    Train neural network models to map single-cell data to IMC populations and predict groups.

    Parameters
    ----------
    adata_sc : ad.AnnData
        Single-cell AnnData object.
    adata_imc : ad.AnnData
        IMC AnnData object.
    gene_protein_mapping : pd.DataFrame
        DataFrame mapping genes to proteins.
    gene_protein_mapping_usecol : str
        Column in gene_protein_mapping indicating which mappings to use.
    gene_protein_mapping_sc_col : str
        Column name for single-cell gene mappings.
    gene_protein_mapping_imc_col : str
        Column name for IMC protein mappings.
    sc_pop_exclude : List[str]
        List of single-cell populations to exclude.
    sc_obs : str
        Observation column for single-cell populations.
    imc_pop_exclude : List[str]
        List of IMC populations to exclude.
    imc_obs : str
        Observation column for IMC populations.
    scale : str
        Scaling method ('ss' for StandardScaler, 'mm' for MinMaxScaler).
    confidence_threshold : float, optional
        Confidence threshold for predictions.
    neurons : int
        Number of neurons in hidden layers.
    optimizer : str
        Optimizer for training the model.
    batch_size : int
        Batch size for training.
    """
    gene_protein_mapping = gene_protein_mapping[gene_protein_mapping[gene_protein_mapping_usecol] == 1]

    # Load and filter single-cell data
    adata_sc = adata_sc[~adata_sc.obs[sc_obs].isin(sc_pop_exclude)].copy()
    X_sc = adata_sc.X[:, [adata_sc.var.index.get_loc(item) for item in gene_protein_mapping[gene_protein_mapping_sc_col]]].toarray()
    y_sc = adata_sc.obs[sc_obs].cat.codes

    # Load and filter IMC data
    adata_imc = adata_imc[~adata_imc.obs[imc_obs].isin(imc_pop_exclude)].copy()
    X_imc = adata_imc.X[:, [adata_imc.var.index.get_loc(item) for item in gene_protein_mapping[gene_protein_mapping_imc_col]]]
    y_imc = adata_imc.obs[imc_obs].cat.codes

    # Train models
    model_imc, codes_imc = train_model(X_imc, y_imc, scale=scale, adata=adata_imc, obs=imc_obs, neurons=neurons, optimizer=optimizer, batch_size=batch_size)
    model_sc, codes_sc = train_model(X_sc, y_sc, scale=scale, adata=adata_sc, obs=sc_obs, neurons=neurons, optimizer=optimizer, batch_size=batch_size)

    # Reload original X for SC data
    X_sc = adata_sc.X[:, [adata_sc.var.index.get_loc(item) for item in gene_protein_mapping[gene_protein_mapping_sc_col]]].toarray()

    # Reload original X for IMC data
    X_imc = adata_imc.X[:, [adata_imc.var.index.get_loc(item) for item in gene_protein_mapping[gene_protein_mapping_imc_col]]]

    # Predict groups and save results
    imc_predicting_sc, predictions_confidence_sc = predict_groups(model_imc, X_sc, codes_imc, scale=scale, confidence_threshold=confidence_threshold)
    pd.DataFrame(zip(imc_predicting_sc, predictions_confidence_sc), columns=['Population', 'Prediction Accuracy']).to_csv('imc_predicting_sc.csv')
    
    sc_predicting_imc, predictions_confidence_imc = predict_groups(model_sc, X_imc, codes_sc, scale=scale, confidence_threshold=confidence_threshold)
    pd.DataFrame(zip(sc_predicting_imc, predictions_confidence_imc), columns=['Population', 'Prediction Accuracy']).to_csv('sc_predicting_imc.csv')
