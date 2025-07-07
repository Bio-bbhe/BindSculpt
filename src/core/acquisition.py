# src/core/acquisition.py
import logging
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans  # For initial diversity
from sklearn.metrics.pairwise import euclidean_distances  # For diversity distance

logger = logging.getLogger(__name__)


class AcquisitionFunction:
    def __init__(self, full_config):
        self.config = full_config
        self.al_config = self.config.active_learning
        self.surrogate_model_config = self.config.models.surrogate_model
        logger.info("Acquisition functions initialized.")

    def select_candidates(self, binders_data_list, batch_size: int = None, strategy: str = None,
                          group_by_key: str = None, num_top: int = None, num_bottom: int = None,
                          num_uncertain: int = None
                          ):
        """
        Selects binders for the next active learning iteration based on the configured strategy.
        Can perform group-based selection or top-N-bottom-M.

        Args:
            binders_data_list (list): A list of dictionaries of binders.
            batch_size (int, optional): Total number of candidates to select (if not top/bottom specific).
            strategy (str, optional): Selection strategy. Defaults to config.
            group_by_key (str, optional): Key to group by for selection (e.g., 'parent_id', 'original_parent_id').
                                          If provided, selection is applied PER GROUP.
            num_top (int, optional): Number of top candidates to select per group/total for 'top_n_bottom_m' strategies.
            num_bottom (int, optional): Number of bottom candidates to select per group/total for 'top_n_bottom_m' strategies.
            num_uncertain (int, optional): Number of Top M uncertain candidates to select per group/total for 'top_n_bottom_m' strategies.

        Returns:
            list: A list of selected binder dictionaries.
        """
        if not binders_data_list:
            logger.warning("No binders provided for acquisition. Returning empty list.")
            return []

        strategy = strategy if strategy is not None else self.al_config.acquisition_strategy
        target_metric = self.surrogate_model_config.target_metric

        df = pd.DataFrame(binders_data_list)

        # Check if required columns exist before proceeding
        if 'predicted_' + target_metric not in df.columns and "top" in strategy:
            logger.warning(
                f"Predicted metric 'predicted_{target_metric}' not found in data for strategy '{strategy}'. Returning empty list.")
            return []

        if group_by_key and group_by_key not in df.columns:
            logger.warning(f"Grouping key '{group_by_key}' not found in data. Proceeding without grouping.")
            group_by_key = None

        selected_df = pd.DataFrame()

        # --- NEW LOGIC FOR COMBINED STRATEGIES ---
        if group_by_key:
            logger.info(f"Applying selection PER GROUP (by '{group_by_key}') using strategy '{strategy}'.")

            if df[group_by_key].isnull().any():
                logger.warning(f"NaN values found in group_by_key '{group_by_key}'. Filtering out such entries.")
                df = df.dropna(subset=[group_by_key])
                if df.empty: return []

            for group_name, group_df in df.groupby(group_by_key):
                if strategy == "top_n_bottom_m_predicted_iptm":
                    num_top = num_top if num_top is not None else self.al_config.num_top_candidates_per_group
                    num_bottom = num_bottom if num_bottom is not None else self.al_config.num_bottom_candidates_per_group
                    selected_group_df = self._select_top_N_bottom_M(group_df, num_top, num_bottom, target_metric)
                    logger.info(f"Mutant selecting strategy is: {strategy}. Selecting metrics is:{target_metric}")
                # --- NEW STRATEGY BLOCK ---
                elif strategy == "top_n_uncertain_m_predicted_iptm":
                    num_top = num_top if num_top is not None else self.al_config.num_top_candidates_per_group
                    num_uncertain = num_uncertain if num_uncertain is not None else self.al_config.num_uncertain_candidates_per_group
                    selected_group_df = self._select_top_N_uncertain_M(group_df, num_top, num_uncertain, target_metric)
                    logger.info(f"Mutant selecting strategy is: {strategy}. Selecting metrics is:{target_metric}")

                # --- NEW STRATEGY BLOCK ---
                elif strategy in ["top_bottom_uncertain_predicted_iptm", "top_bottom_uncertain_predicted_score"]:
                    num_top = num_top if num_top is not None else self.al_config.num_top_candidates_per_group
                    num_bottom = num_bottom if num_bottom is not None else self.al_config.num_bottom_candidates_per_group
                    num_uncertain = num_uncertain if num_uncertain is not None else self.al_config.num_uncertain_candidates_per_group
                    selected_group_df = self._select_top_bottom_uncertain(group_df, num_top, num_bottom, num_uncertain, target_metric)
                    logger.info(f"Mutant selecting strategy is: {strategy}. Selecting metrics is:{target_metric}")

                else:  # Fallback to existing strategies or random
                    if strategy == "top_iptm":
                        if target_metric not in group_df.columns or group_df[target_metric].isnull().all():
                            logger.warning(
                                f"True metric '{target_metric}' not available for group '{group_name}'. Skipping.")
                            continue
                        selected_group_df = group_df.sort_values(by=target_metric, ascending=False).head(batch_size)
                    elif strategy == "combined_score":
                        if target_metric not in group_df.columns or group_df[target_metric].isnull().all():
                            logger.warning(
                                f"True metric '{target_metric}' not available for group '{group_name}'. Skipping.")
                            continue
                        selected_group_df = group_df.sort_values(by=target_metric, ascending=False).head(batch_size)
                    else:  # Fallback to random if strategy is unknown for group
                        selected_group_df = self._select_random(group_df, batch_size)

                selected_df = pd.concat([selected_df, selected_group_df])
                logger.debug(f"  Selected {len(selected_group_df)} from group '{group_name}'.")

            logger.info(f"Total selected {len(selected_df)} candidates across all groups.")
            return selected_df.to_dict(orient='records')

        else:  # No grouping logic (for parent selection or simple combined pool selection)
            # ... (this part can be simplified or kept for backwards compatibility) ...
            if strategy in ["top_iptm", 'combined_score']:  # Used for parent selection, based on true IPTM
                batch_size = batch_size if batch_size is not None else self.al_config.num_parent_binders_to_optimize
                logger.info(f"Selecting {batch_size} candidates from combined pool using strategy '{strategy}'.")
                if target_metric not in df.columns or df[target_metric].isnull().all():
                    logger.warning(
                        f"True metric '{target_metric}' not available for 'top_iptm' strategy. Falling back to random selection.")
                    return self._select_random(df, batch_size).to_dict(orient='records')
                selected_df = df.sort_values(by=target_metric, ascending=False).head(batch_size)
                logger.info(f"Selected {len(selected_df)} candidates from combined pool using '{strategy}' strategy.")
                return selected_df.to_dict(orient='records')
            else:
                logger.warning(f"Unknown or non-grouped strategy '{strategy}' not implemented. Returning empty list.")
                return []

    def _select_random(self, df, batch_size):
        """Helper to select random candidates from a DataFrame."""
        num_to_select = min(batch_size, len(df))
        if num_to_select == 0: return pd.DataFrame()  # Return empty DataFrame
        selected_df = df.sample(n=num_to_select, random_state=42)
        logger.info(f"Selected {len(selected_df)} candidates randomly.")
        return selected_df

    def _select_top_N_bottom_M(self, df: pd.DataFrame, num_top: int, num_bottom: int, metric_col: str):
        """
        Helper to select top N and bottom M candidates from a DataFrame.
        Assumes higher metric is better for 'top'.
        """
        if df.empty:
            return pd.DataFrame()

        predicted_metric_col = f'predicted_{metric_col}'
        if predicted_metric_col not in df.columns or df[predicted_metric_col].isnull().all():
            logger.warning(
                f"Predicted metric '{predicted_metric_col}' not available for top/bottom selection. Falling back to random selection for this group.")
            return self._select_random(df, num_top + num_bottom)  # Randomly pick total if metric is missing

        # Sort by predicted score
        df_sorted = df.sort_values(by=predicted_metric_col, ascending=False).reset_index(drop=True)

        # Select top N
        top_candidates = df_sorted.head(num_top)

        # Select bottom M
        # Ensure there are enough remaining candidates for bottom selection after taking top
        if num_bottom > 0 and len(df_sorted) > num_top:
            # Exclude top N candidates from consideration for bottom M
            bottom_candidates = df_sorted.tail(num_bottom)
            # It's possible top and bottom overlap if df is very small, or bottom selected from end of remaining
            # To ensure strict separation:
            remaining_df = df_sorted.iloc[num_top:]  # Candidates not in top N
            if len(remaining_df) > num_bottom:
                bottom_candidates = remaining_df.sample(n=num_bottom, random_state=42)  # Random from bottom pool
            else:
                bottom_candidates = remaining_df  # Take all remaining if less than num_bottom
        else:
            bottom_candidates = pd.DataFrame()  # Empty DataFrame if no bottom needed/possible

        # Combine top and bottom candidates
        selected_df = pd.concat([top_candidates, bottom_candidates]).drop_duplicates(subset=['id'])  # Ensure unique IDs

        logger.debug(
            f"Selected {len(top_candidates)} top and {len(bottom_candidates)} bottom candidates. Total: {len(selected_df)}")
        return selected_df

    def select_top_N_diverse_from_pool(self, binders_data_list: list, num_to_select: int, num_diversity_groups: int,
                                       metric_for_selection: str, data_manager):
        """
        Selects a diverse set of top-performing binders from a large pool.
        This is for initial generation.

        Args:
            binders_data_list (list): List of dictionaries, each containing
                                      'id', 'sequence', 'embedding', and the metric_for_selection.
            num_to_select (int): The target number of diverse candidates to select.
            num_diversity_groups (int): Number of clusters to form for diversity.
            metric_for_selection (str): The metric to use for ranking within clusters (e.g., 'iptm').
            data_manager (DataManager): Instance of DataManager to load embeddings if needed.

        Returns:
            list: A list of selected, diverse binder dictionaries.
        """
        if not binders_data_list:
            logger.warning("No binders provided for diverse selection. Returning empty list.")
            return []

        # Ensure embeddings are loaded for clustering
        # The data_manager.load_embeddings_for_binders might already have loaded them, but ensure
        binders_data_list_with_embeddings = data_manager.load_embeddings_for_binders(binders_data_list)

        # Filter for items with valid embeddings and metric
        eligible_binders = [
            b for b in binders_data_list_with_embeddings
            if b.get('embedding') is not None and b.get(metric_for_selection) is not None
        ]
        if not eligible_binders:
            logger.warning("No eligible binders with embeddings or metric for diverse selection. Returning empty list.")
            return []

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(eligible_binders)

        # Ensure target metric is numeric
        df[metric_for_selection] = pd.to_numeric(df[metric_for_selection], errors='coerce')
        df = df.dropna(subset=[metric_for_selection]).reset_index(drop=True)

        if df.empty:
            logger.warning("No binders remaining after metric conversion/NA drop for diverse selection.")
            return []

        # Perform clustering on embeddings
        embeddings = np.vstack(df['embedding'].values)  # Stack embeddings into a 2D array

        if len(embeddings) < num_diversity_groups:
            logger.warning(
                f"Number of binders ({len(embeddings)}) is less than desired diversity groups ({num_diversity_groups}). Setting groups to binder count.")
            num_diversity_groups = len(embeddings)
            if num_diversity_groups == 0: return []

        # Use MiniBatchKMeans for efficiency with potentially large number of binders
        kmeans = MiniBatchKMeans(n_clusters=num_diversity_groups, random_state=42, n_init='auto')
        df['cluster'] = kmeans.fit_predict(embeddings)

        selected_diverse_binders = []

        # Select the best performing binder from each cluster
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster_id]
            # Sort by metric (highest is better)
            best_in_cluster = cluster_df.sort_values(by=metric_for_selection, ascending=False).iloc[0]
            selected_diverse_binders.append(best_in_cluster.to_dict())

        # If we selected more than num_to_select (unlikely with this strategy but possible if clusters are large and unique counts are high),
        # or if we selected fewer, we need to adjust.
        # If we selected fewer than num_to_select, fill with next best overall.
        # If we selected more, sort by metric and take top.

        if len(selected_diverse_binders) > num_to_select:
            selected_diverse_binders.sort(key=lambda x: x[metric_for_selection], reverse=True)
            selected_diverse_binders = selected_diverse_binders[:num_to_select]
        elif len(selected_diverse_binders) < num_to_select and len(df) > len(selected_diverse_binders):
            logger.info(
                f"Selected {len(selected_diverse_binders)} unique clusters, but need {num_to_select}. Filling remaining with top overall.")
            # Get IDs of already selected binders
            selected_ids = {b['id'] for b in selected_diverse_binders}
            # Sort all eligible binders and take remaining top ones that are not already selected
            remaining_eligible_binders = df[~df['id'].isin(selected_ids)].sort_values(by=metric_for_selection,
                                                                                      ascending=False).to_dict(
                orient='records')

            selected_diverse_binders.extend(remaining_eligible_binders[:num_to_select - len(selected_diverse_binders)])

        logger.info(f"Final diverse selection yielded {len(selected_diverse_binders)} binders.")
        return selected_diverse_binders

    def _select_top_N_uncertain_M(self, df: pd.DataFrame, num_top: int, num_uncertain: int, metric_col: str):
        """
        NEW HELPER: Selects top N predicted candidates and top M uncertain candidates.
        """
        if df.empty:
            return pd.DataFrame()

        predicted_metric_col = f'predicted_{metric_col}'
        uncertainty_col = 'prediction_uncertainty'

        # Check for required columns
        if predicted_metric_col not in df.columns or uncertainty_col not in df.columns:
            logger.warning(
                f"Missing '{predicted_metric_col}' or '{uncertainty_col}' for top/uncertain selection. Falling back to random selection for this group.")
            return self._select_random(df, num_top + num_uncertain)

        # Ensure columns are numeric and handle NaNs
        df_clean = df.dropna(subset=[predicted_metric_col, uncertainty_col]).copy()
        if df_clean.empty:
            logger.warning(
                "No valid candidates with predicted scores and uncertainty. Returning empty list for this group.")
            return pd.DataFrame()

        # Select top N candidates (exploitation)
        top_candidates = df_clean.nlargest(num_top, predicted_metric_col)

        # Exclude the top candidates from the uncertainty pool to avoid selecting the same ones twice
        remaining_candidates = df_clean.drop(top_candidates.index)

        if remaining_candidates.empty:  # If top selection exhausted all candidates
            logger.warning("No remaining candidates for uncertainty selection after selecting top N.")
            return top_candidates  # Just return the top candidates

        # Select top M uncertain candidates from the remaining pool (exploration)
        uncertain_candidates = remaining_candidates.nlargest(num_uncertain, uncertainty_col)

        # Combine the two sets
        selected_df = pd.concat([top_candidates, uncertain_candidates])

        logger.debug(
            f"Selected {len(top_candidates)} top-predicted and {len(uncertain_candidates)} most-uncertain candidates. Total: {len(selected_df)}")
        return selected_df

        # --- NEW HELPER METHOD ---

    def _select_top_bottom_uncertain(self, df: pd.DataFrame, num_top: int, num_bottom: int, num_uncertain: int,
                                     metric_col: str):
        """
        NEW HELPER: Selects top N predicted, bottom M predicted, and top K uncertain candidates.
        """
        if df.empty:
            return pd.DataFrame()

        predicted_metric_col = f'predicted_{metric_col}'
        uncertainty_col = 'prediction_uncertainty'

        # Check for required columns
        if predicted_metric_col not in df.columns or uncertainty_col not in df.columns:
            logger.warning(
                f"Missing required columns for top/bottom/uncertain selection. Falling back to random selection.")
            return self._select_random(df, num_top + num_bottom + num_uncertain)

        df_clean = df.dropna(subset=[predicted_metric_col, uncertainty_col]).copy()
        if df_clean.empty:
            logger.warning("No valid candidates with predicted scores and uncertainty. Returning empty list.")
            return pd.DataFrame()

        # 1. Select top N candidates (exploitation)
        top_candidates = df_clean.nlargest(num_top, predicted_metric_col)

        # 2. Exclude top candidates from the remaining pool
        remaining_candidates_after_top = df_clean.drop(top_candidates.index)
        if remaining_candidates_after_top.empty:
            logger.warning("No candidates left after selecting top N. Returning only top candidates.")
            return top_candidates

        # 3. Select bottom M candidates from the remaining pool (boundary refinement)
        bottom_candidates = remaining_candidates_after_top.nsmallest(num_bottom, predicted_metric_col)

        # 4. Exclude bottom candidates from the final pool for uncertainty selection
        remaining_candidates_after_bottom = remaining_candidates_after_top.drop(bottom_candidates.index)
        if remaining_candidates_after_bottom.empty:
            logger.warning("No candidates left after selecting top and bottom. Returning top + bottom.")
            return pd.concat([top_candidates, bottom_candidates])

        # 5. Select top K uncertain candidates from the final remaining pool (exploration)
        uncertain_candidates = remaining_candidates_after_bottom.nlargest(num_uncertain, uncertainty_col)

        # Combine the three sets
        selected_df = pd.concat([top_candidates, bottom_candidates, uncertain_candidates]).drop_duplicates(
            subset=['id'])

        logger.debug(
            f"Selected {len(top_candidates)} top, {len(bottom_candidates)} bottom, and {len(uncertain_candidates)} uncertain. Total: {len(selected_df)}")
        return selected_df
