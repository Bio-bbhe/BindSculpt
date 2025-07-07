# src/ml_models/surrogate_model.py
import logging
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # For saving/loading models

logger = logging.getLogger(__name__)


class SurrogateModel:
    def __init__(self, config):
        self.config = config.models.surrogate_model  # New config section for Surrogate Model
        self.model = None  # RandomForestRegressor instance
        self.target_metric = self.config.target_metric  # e.g., 'iptm' or 'ptm'
        logger.info(f"SurrogateModel initialized. Target metric for prediction: {self.target_metric}")

    def train(self, evaluated_binders_data: list, iteration_num: int):
        """
        Trains the Random Forest Regressor model.

        Args:
            evaluated_binders_data (list): List of dictionaries, each containing
                                           'embedding' (np.array) and the target metric (e.g., 'iptm').
            iteration_num (int): iteration number.
        """
        if not evaluated_binders_data:
            logger.warning("No data provided for surrogate model training. Skipping training.")
            return

        # Filter data to only include valid entries for training
        train_data = [
            b for b in evaluated_binders_data
            if 'embedding' in b and b['embedding'] is not None and
               self.target_metric in b and b[self.target_metric] is not None
        ]
        if not train_data:
            logger.warning("No valid data points (embeddings or target metric) for training. Skipping training.")
            return

        X_train = np.array([item['embedding'] for item in train_data])
        y_train = np.array([item[self.target_metric] for item in train_data])

        if len(X_train) == 0:
            logger.warning("No features (embeddings) found for training. Skipping training.")
            return
        if len(y_train) == 0:
            logger.warning("No labels (target metric) found for training. Skipping training.")
            return

        logger.info(f"Training Surrogate Model with {len(X_train)} data points.")

        # Initialize RandomForestRegressor with parameters from config
        rf_params = self.config.params
        self.model = RandomForestRegressor(
            n_estimators=rf_params.get('n_estimators', 100),
            criterion=rf_params.get('criterion', 'friedman_mse'),
            max_depth=rf_params.get('max_depth', None),
            min_samples_split=rf_params.get('min_samples_split', 2),
            min_samples_leaf=rf_params.get('min_samples_leaf', 1),
            max_features=rf_params.get('max_features', 1.0),
            random_state=rf_params.get('random_state', 1),
            min_impurity_decrease=rf_params.get('min_impurity_decrease', 0.0),
            bootstrap=rf_params.get('bootstrap', True),
            oob_score=rf_params.get('oob_score', False),
            verbose=rf_params.get('verbose', False),
            warm_start=rf_params.get('warm_start', False),
            ccp_alpha=rf_params.get('ccp_alpha', 0.0),
            n_jobs=rf_params.get('n_jobs', -1)  # Use all available cores by default
        )

        self.model.fit(X_train, y_train)

        # Evaluate on training data
        y_pred_train = self.model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred_train)
        r2 = r2_score(y_train, y_pred_train)
        logger.info(f"Surrogate Model Training Results: MSE={mse:.4f}, R2={r2:.4f}")

        # Optional: Save the trained model
        if self.config.get('save_model', False):
            save_dir = self.config.get('save_dir')
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                model_save_path = os.path.join(save_dir, f"surrogate_model_iter_{iteration_num}.joblib")
                joblib.dump(self.model, model_save_path)
                logger.info(f"Surrogate model saved to {model_save_path}")

    def predict(self, candidate_embeddings_data: list):
        """
        Predicts the target metric for new candidate embeddings.

        Args:
            candidate_embeddings_data (list): List of dictionaries, each containing
                                              'id', 'sequence', and 'embedding' (np.array).

        Returns:
            list: The input list of dictionaries, with 'predicted_metric' added to each.
        """
        if self.model is None:
            logger.error("Surrogate model not trained. Cannot make predictions. Returning original data.")
            return candidate_embeddings_data
        if not candidate_embeddings_data:
            logger.warning("No candidate embeddings provided for prediction. Returning empty list.")
            return []

        X_predict = np.array([item['embedding'] for item in candidate_embeddings_data])
        if len(X_predict) == 0:
            logger.warning("No features (embeddings) found for prediction. Returning original data.")
            return candidate_embeddings_data

        logger.info(f"Predicting {self.target_metric} for {len(X_predict)} candidates.")
        y_pred = self.model.predict(X_predict)

        # Add predictions back to the original data
        for i, item in enumerate(candidate_embeddings_data):
            item[f'predicted_{self.target_metric}'] = y_pred[i]

        logger.info("Prediction complete.")
        return candidate_embeddings_data

    def predict_with_uncertainty(self, candidate_embeddings_data: list):
        """
        NEW METHOD: Predicts the target metric and also an uncertainty score
        (standard deviation of individual tree predictions).
        """
        if self.model is None:
            logger.error("Surrogate model not trained. Cannot make predictions. Returning original data.")
            return candidate_embeddings_data
        if not candidate_embeddings_data:
            logger.warning("No candidate embeddings provided for uncertainty prediction. Returning empty list.")
            return []

        X_predict = np.array([item['embedding'] for item in candidate_embeddings_data])
        if len(X_predict) == 0:
            logger.warning("No features (embeddings) found for uncertainty prediction. Returning original data.")
            return candidate_embeddings_data

        logger.info(f"Predicting {self.target_metric} with uncertainty for {len(X_predict)} candidates.")

        # Get predictions from each individual tree in the forest
        # This returns a list of arrays, one array per tree
        tree_predictions = [tree.predict(X_predict) for tree in self.model.estimators_]

        # Stack predictions into a single array: shape (n_estimators, n_samples)
        stacked_predictions = np.stack(tree_predictions, axis=0)

        # Calculate the mean (the final prediction) and std dev (the uncertainty)
        # along the tree axis (axis=0)
        mean_predictions = np.mean(stacked_predictions, axis=0)
        std_dev_predictions = np.std(stacked_predictions, axis=0)

        # Add predictions and uncertainty back to the original data
        for i, item in enumerate(candidate_embeddings_data):
            item[f'predicted_{self.target_metric}'] = mean_predictions[i]
            item[f'prediction_uncertainty'] = std_dev_predictions[i]

        logger.info("Prediction with uncertainty complete.")
        return candidate_embeddings_data