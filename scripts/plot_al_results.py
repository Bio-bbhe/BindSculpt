# scripts/plot_al_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import logging
import math

logger = logging.getLogger(__name__)
# Ensure logger is configured to show INFO messages if running this standalone
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def _load_and_filter_data(al_state_csv_path: str):
    """
    Helper function to load the AL state CSV and perform common filtering.
    """
    if not os.path.exists(al_state_csv_path):
        logger.error(f"AL state CSV file not found at: {al_state_csv_path}")
        raise FileNotFoundError(f"CSV not found: {al_state_csv_path}")

    # Define dtypes for all relevant columns to avoid DtypeWarning and ensure consistency
    # Add any other columns from your CSV here with their expected types
    expected_dtypes = {
        'id': str, 'parent_id': str, 'original_parent_id': str,
        'embedding_path': str, 'iteration': pd.Int64Dtype(),
        'iptm': float, 'ptm': float, 'ranking_score': float,
        'af3_status': str, 'binder_sequence': str, 'target_id': str,
        'mutation_type': str, 'original_aa_pos_mut': str,
        'mutations_applied': str, 'mutation_history': str,
        'target_sequence': str,  # From DTypeWarning
        'sampling_strategy': str,  # From DTypeWarning
        'generation_params_str': str,  # From DTypeWarning
        'af3_pdb_path': str,  # From DTypeWarning
        'af3_output_dir': str,  # From DTypeWarning
        'filter_reason': str,  # From DTypeWarning
        'embedding': str,  # Assuming string representation, adjust if it's path or other
        # Add other columns mentioned in DtypeWarning if they exist and you know their type
        # 'generation_time_s': float, # Example
        # 'af3_error': str, # Example
        # 'af3_prediction_time_s': float, # Example
        # 'passed_filters': bool, # Example if it's boolean-like
        # 'predicted_iptm': float # Example
    }

    try:
        df = pd.read_csv(
            al_state_csv_path,
            dtype=expected_dtypes,
            keep_default_na=True,
            na_values=['', 'NA', 'N/A', 'NaN', 'none', 'None'],  # Treat various NaN representations
            low_memory=False  # Can help with mixed types if dtypes aren't perfectly exhaustive
        )
        logger.info(f"Loaded {len(df)} entries from {al_state_csv_path}")
        if 'original_parent_id' in df.columns:
            logger.info(
                f"Sample original_parent_id values from raw CSV: {df['original_parent_id'].dropna().unique()[:5]}")
        filtered_df = df[(df['iptm'] >= 0.65) & (df['ptm'] >= 0.65)]
        print(len(filtered_df))
        return df

    except pd.errors.EmptyDataError:
        logger.warning(f"AL state CSV file at {al_state_csv_path} is empty. No data to plot.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An error occurred while loading or preprocessing CSV: {e}")
        raise


def plot_al_line_trajectory(
        df: pd.DataFrame,
        output_plot_path: str,
        title: str = "Active Learning Trajectory",
        ref_line_value: float = 1.0
):
    df_evaluated = df[
        (df['af3_status'] == 'success') &
        (df['iptm'].notna()) &
        (df['ptm'].notna()) &
        (df['iteration'].notna())
        ].copy()

    if df_evaluated.empty:
        logger.warning("No successful AF3 evaluations found for line plot. Plot not generated.")
        return

    metrics_summary = df_evaluated.groupby('iteration').agg(
        mean_iptm=('iptm', 'mean'),
        max_iptm=('iptm', 'max'),
        mean_ptm=('ptm', 'mean'),
        max_ptm=('ptm', 'max'),
        count=('id', 'size')
    ).reset_index()

    logger.info(f"Summary metrics per iteration for line plot:\n{metrics_summary}")

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.lineplot(data=metrics_summary, x='iteration', y='mean_iptm', marker='o', label='Mean IPTM', color='skyblue')
    sns.lineplot(data=metrics_summary, x='iteration', y='max_iptm', marker='x', label='Max IPTM', color='darkblue')
    if ref_line_value is not None: plt.axhline(y=ref_line_value, color='black', linestyle=':', linewidth=1.5)
    plt.title('IPTM Score Trajectory')
    plt.xlabel('Active Learning Iteration')
    plt.ylabel('IPTM Score')
    plt.ylim(0, 1)
    if not metrics_summary.empty:
        plt.xticks(sorted(metrics_summary['iteration'].unique().astype(int)))
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sns.lineplot(data=metrics_summary, x='iteration', y='mean_ptm', marker='o', label='Mean PTM', color='lightcoral')
    sns.lineplot(data=metrics_summary, x='iteration', y='max_ptm', marker='x', label='Max PTM', color='darkred')
    if ref_line_value is not None: plt.axhline(y=ref_line_value, color='black', linestyle=':', linewidth=1.5)
    plt.title('PTM Score Trajectory')
    plt.xlabel('Active Learning Iteration')
    plt.ylabel('PTM Score')
    plt.ylim(0, 1)
    if not metrics_summary.empty:
        plt.xticks(sorted(metrics_summary['iteration'].unique().astype(int)))
    plt.legend()
    plt.grid(True)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(output_plot_path, dpi=300)
    logger.info(f"Line plot saved to {output_plot_path}")
    plt.close()


def plot_al_violin_trajectory(
        df: pd.DataFrame,
        output_plot_path: str,
        metric_column: str = 'iptm',
        title: str = "Active Learning Trajectory of AF3 Metrics",
        y_label: str = "Score",
        reference_line_value: float = 1.0,
        color_split_iteration: int = 3,
        highlight_parents: list = None,
        color_by_original_parent: bool = False,  # Default is now True via argparse if not explicitly set to False
        parent_color_map: dict = None
):
    if highlight_parents is None:
        highlight_parents = []

    df_evaluated = df[
        (df['af3_status'] == 'success') &
        (df[metric_column].notna())
        ].copy()

    if df_evaluated.empty:
        logger.warning(
            f"No successful AF3 evaluations with valid {metric_column} found for violin plot. Plot not generated.")
        return

    if 'iteration' in df_evaluated.columns:
        df_evaluated['iteration'] = df_evaluated['iteration'].astype(int)

    df_evaluated['Phase'] = df_evaluated['iteration'].apply(
        lambda x: 'Early' if x <= color_split_iteration else 'Late'
    )
    violin_palette = {'Early': 'tab:orange', 'Late': 'tab:blue'}

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    violin_ax = sns.violinplot(
        x='iteration',
        y=metric_column,
        hue='Phase',
        data=df_evaluated,
        palette=violin_palette,
        inner='quartile',
        linewidth=1.0,
        edgecolor='black',
        cut=0,
    )

    logger.info("--- Stripplot Parameter Setup Diagnostics ---")
    logger.info(f"color_by_original_parent flag: {color_by_original_parent}")
    logger.info(f"highlight_parents argument: {highlight_parents}")
    logger.info(f"parent_color_map provided: {'Yes' if parent_color_map else 'No'}")
    if parent_color_map:
        logger.info(f"parent_color_map content (first 3 items): {list(parent_color_map.items())[:3]}")

    original_parent_id_column_present = 'original_parent_id' in df_evaluated.columns
    logger.info(f"'original_parent_id' column present in df_evaluated: {original_parent_id_column_present}")

    num_unique_parents = 0
    if original_parent_id_column_present:
        num_unique_parents = df_evaluated['original_parent_id'].nunique()
        logger.info(f"Number of unique non-NaN 'original_parent_id' values: {num_unique_parents}")
        logger.info(
            f"Value counts for 'original_parent_id' (top 5, incl. NaN):\n{df_evaluated['original_parent_id'].value_counts(dropna=False).head()}")

    final_s_hue = None
    final_s_palette = None
    final_s_color_param = 'gray'
    final_s_edgecolor = 'black'
    final_s_size = 7
    final_s_linewidth = 0.5
    stripplot_dodge = True

    if color_by_original_parent and original_parent_id_column_present:
        logger.info("Attempting to color by 'original_parent_id'.")
        if num_unique_parents > 0:
            final_s_hue = 'original_parent_id'

            # --- REVISED PALETTE LOGIC ---
            palette_set_from_custom_map = False
            if parent_color_map:
                logger.info("Checking custom parent_color_map for stripplot palette.")
                if isinstance(parent_color_map, dict) and parent_color_map:
                    final_s_palette = parent_color_map
                    palette_set_from_custom_map = True
                    logger.info("Successfully set final_s_palette from custom parent_color_map.")

                    unique_parents_in_data = df_evaluated['original_parent_id'].dropna().unique()
                    missing_keys = [pid for pid in unique_parents_in_data if pid not in parent_color_map]
                    if missing_keys:
                        logger.warning(f"Custom parent_color_map is missing colors for parent IDs: {missing_keys}. "
                                       "Seaborn will handle these with default colors or raise an error.")
                else:
                    logger.warning(
                        "parent_color_map was provided but is not a valid non-empty dictionary. Falling back to default palettes.")

            if not palette_set_from_custom_map:
                logger.info("Custom map not used or invalid; generating default palette.")
                if num_unique_parents > 20:
                    logger.warning(
                        f"Too many ({num_unique_parents}) unique original_parent_ids for distinct coloring. Some colors may repeat. Using 'husl'.")
                    final_s_palette = sns.color_palette("husl", num_unique_parents)
                elif num_unique_parents == 1:
                    logger.info("Only one unique original_parent_id. Using 'tab10' (1 color).")
                    final_s_palette = sns.color_palette("tab10", 1)
                elif num_unique_parents > 1:  # 2 to 20
                    ordered_parent_ids = sorted(df_evaluated['original_parent_id'].dropna().unique())
                    default_colors = sns.color_palette("tab20", num_unique_parents)
                    final_s_palette = {pid: color for pid, color in zip(ordered_parent_ids, default_colors)}
                    logger.info(f"Using default 'tab20' palette mapped to parent IDs.")
                else:  # num_unique_parents == 0
                    logger.warning("No unique parent IDs found for default palette generation.")
                    final_s_palette = None

            final_s_color_param = None
            logger.info(f"Coloring points by 'original_parent_id'. hue='{final_s_hue}', color=None.")
        else:
            logger.warning(
                "Coloring by 'original_parent_id' selected, but no valid unique parent IDs found. Points will be default gray."
            )
            final_s_hue = None
            final_s_palette = None
            final_s_color_param = 'lightgray'
    elif highlight_parents:
        logger.info("Highlighting specific parents. Base points will be white.")
        final_s_hue = None
        final_s_color_param = 'white'
    else:
        logger.info("Default point styling: white points, potentially dodged by Phase.")
        final_s_hue = 'Phase'
        final_s_palette = {'Early': 'white', 'Late': 'white'}
        final_s_color_param = None

    logger.info(
        f"Final stripplot params: hue={final_s_hue}, color_param={final_s_color_param}, dodge={stripplot_dodge}")
    if final_s_palette:
        if isinstance(final_s_palette, dict):
            logger.info(
                f"Palette type: <class 'dict'>, first few items: {list(final_s_palette.items())[:min(len(final_s_palette), 3)]}")
        elif isinstance(final_s_palette, list):
            logger.info(
                f"Palette type: <class 'list'> (Seaborn _ColorPalette), first few colors: {final_s_palette[:min(len(final_s_palette), 3)]}")
        else:
            logger.info(f"Palette type: {type(final_s_palette)}")

    stripplot_ax = sns.stripplot(
        x='iteration',
        y=metric_column,
        hue=final_s_hue,
        data=df_evaluated,
        palette=final_s_palette,
        color=final_s_color_param,
        edgecolor=final_s_edgecolor,
        linewidth=final_s_linewidth,
        jitter=0.2,
        size=final_s_size,
        dodge=stripplot_dodge,
        ax=violin_ax,
        zorder=1
    )

    if highlight_parents and not color_by_original_parent:  # Only highlight if not already coloring all by parent
        highlight_palette_specific = sns.color_palette("Dark2", len(highlight_parents))
        highlight_color_map = {parent_id: color for parent_id, color in
                               zip(highlight_parents, highlight_palette_specific)}
        df_highlight = df_evaluated[df_evaluated['original_parent_id'].isin(highlight_parents)].copy()
        if not df_highlight.empty:
            logger.info(f"Highlighting specific data points for parents: {highlight_parents}")
            for i, parent_id_to_highlight in enumerate(highlight_parents):
                parent_df_to_highlight = df_highlight[df_highlight['original_parent_id'] == parent_id_to_highlight]
                if not parent_df_to_highlight.empty:
                    sns.stripplot(
                        x='iteration', y=metric_column, data=parent_df_to_highlight,
                        color=highlight_color_map[parent_id_to_highlight],
                        edgecolor='black', linewidth=1.0, jitter=0.2, size=final_s_size + 2,
                        dodge=stripplot_dodge, ax=violin_ax, zorder=2,
                        label=f"Parent: {parent_id_to_highlight}"
                    )
        else:
            logger.warning(f"No data for explicitly highlighted parents: {highlight_parents}.")

    if reference_line_value is not None: plt.axhline(y=reference_line_value, color='black', linestyle=':',
                                                     linewidth=1.5)
    plt.title(title, fontsize=16)
    plt.xlabel('Active Learning Iteration', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.ylim(0.0, 1.0)
    if not df_evaluated['iteration'].empty:
        unique_iterations = sorted(df_evaluated['iteration'].unique())
        plt.xticks(unique_iterations)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_legend_dict = {}
    phase_labels = ['Early', 'Late']
    for handle, label in zip(handles, labels):
        if label in phase_labels and label not in unique_legend_dict:
            unique_legend_dict[label] = handle
    for handle, label in zip(handles, labels):
        if label not in unique_legend_dict:
            unique_legend_dict[label] = handle

    def legend_sort_key(label_item_tuple):
        label = label_item_tuple[0]
        if label == 'Early': return (0, label)
        if label == 'Late': return (1, label)
        return (2, label)

    sorted_legend_items = sorted(unique_legend_dict.items(), key=legend_sort_key)

    if sorted_legend_items:
        sorted_handles = [item[1] for item in sorted_legend_items]
        sorted_labels = [item[0] for item in sorted_legend_items]
        plt.legend(sorted_handles, sorted_labels, title="Legend", loc='upper left', bbox_to_anchor=(1.02, 1),
                   title_fontsize='medium', fontsize='small')
        plt.tight_layout(rect=[0, 0, 0.83, 1])
    else:
        current_legend = plt.gca().get_legend()
        if current_legend: current_legend.remove()
        plt.tight_layout()

    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Violin plot saved to {output_plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate various plots for Active Learning trajectory.")
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to the al_state.csv file.')
    parser.add_argument('--output_dir', type=str, default='plots/',
                        help='Directory to save output plots.')
    parser.add_argument('--plot_type', type=str, choices=['line', 'violin', 'all'], default='all',
                        help='Type of plot to generate.')
    parser.add_argument('--metric', type=str, default='iptm',
                        help='Metric for violin plot (e.g., "iptm", "ptm").')
    parser.add_argument('--title', type=str, default='BindSculpt Active Learning Trajectory',
                        help='Main title for the plots.')
    parser.add_argument('--y_label', type=str, default='Score',
                        help='Label for the Y-axis (primarily for violin plot).')
    parser.add_argument('--ref_line_value', type=str, default="0.8",  # String to allow "None"
                        help='Value for the horizontal reference line. Set to "None" (case-insensitive) to disable.')
    parser.add_argument('--color_split_iter', type=int, default=3,
                        help='Iteration for Early/Late phase split in violin plot.')
    parser.add_argument('--highlight_parents', type=str, nargs='*', default=None,
                        help='List of original_parent_ids to highlight.')
    parser.add_argument(
        '--no-parent-coloring',
        action='store_false',
        dest='color_by_original_parent',
        help='Disable coloring data points by original_parent_id (default is to color by parent).'
    )
    # Set default for color_by_original_parent to True if the flag --no-parent-coloring is NOT present
    parser.set_defaults(color_by_original_parent=True)

    args = parser.parse_args()

    logger.info(f"Parsed arguments: {args}")  # Log parsed arguments

    ref_line_value_parsed = None
    if args.ref_line_value.strip().lower() != 'none':
        try:
            ref_line_value_parsed = float(args.ref_line_value)
        except ValueError:
            logger.warning(f"Invalid ref_line_value '{args.ref_line_value}'. Disabling reference line.")
    else:
        logger.info("Reference line disabled by user ('None').")

    if not logging.getLogger(__name__).hasHandlers() and not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        df_full = _load_and_filter_data(args.input_csv)
        if df_full.empty:
            logger.error("No data available after loading/filtering. Exiting plotting script.")
            return

        if 'iteration' in df_full.columns:
            try:
                df_full['iteration'] = pd.to_numeric(df_full['iteration'], errors='coerce').astype(pd.Int64Dtype())
            except Exception as e:
                logger.warning(f"Could not convert 'iteration' column to integer: {e}. Plotting may be affected.")

        if args.plot_type == 'line' or args.plot_type == 'all':
            plot_title_line = f"{args.title} ({args.metric.upper()} Line Trajectory)"
            output_line_plot_path = os.path.join(args.output_dir, f"{args.metric}_line_trajectory.svg")
            plot_al_line_trajectory(
                df=df_full,
                output_plot_path=output_line_plot_path,
                title=plot_title_line,
                ref_line_value=ref_line_value_parsed
            )

        if args.plot_type == 'violin' or args.plot_type == 'all':
            violin_y_label = f"{args.metric.upper()} Score" if args.y_label == 'Score' else args.y_label
            plot_title_violin = f"{args.title} ({violin_y_label} Distribution)"
            output_violin_plot_path = os.path.join(args.output_dir, f"{args.metric}_violin_distribution.svg")

            current_highlight_parents = args.highlight_parents if args.highlight_parents is not None else []

            # DEFINE YOUR CUSTOM COLOR MAP HERE
            # Ensure keys match exactly with values in your 'original_parent_id' column
            my_custom_colors_map = {
                'binder_1749049094_0': 'dodgerblue',  # A specific blue
                'binder_1749049094_1': 'cyan',  # A specific orange/red
                'binder_1749049094_2': '#00FF00',  # A specific green
                'binder_1749049094_3': 'red',  # Yellow/gold
                'binder_1749049094_4': (0.5, 0.2, 0.8),  # mediumpurple A purple
                'binder_1749049094_5': '#4DAF4A',  # A brown
                # Add/modify as needed for your actual parent IDs
            }
            # Pass the custom map only if coloring by parent is active
            active_parent_color_map = my_custom_colors_map if args.color_by_original_parent else None

            plot_al_violin_trajectory(
                df=df_full,
                output_plot_path=output_violin_plot_path,
                metric_column=args.metric,
                title=plot_title_violin,
                y_label=violin_y_label,
                reference_line_value=ref_line_value_parsed,
                color_split_iteration=args.color_split_iter,
                highlight_parents=current_highlight_parents,
                color_by_original_parent=False, # args.color_by_original_parent,
                parent_color_map=active_parent_color_map  # Pass the map
            )

    except FileNotFoundError:
        pass
    except Exception as e:
        logger.critical(f"A critical error occurred in the plotting script: {e}", exc_info=True)


if __name__ == "__main__":
    main()

# DEFINE YOUR CUSTOM COLOR MAP HERE
my_custom_colors = {
    'binder_1749049094_0': 'red',
    'binder_1749049094_1': 'red',  # '#00FF00',  # Bright Green
    'binder_1749049094_2': 'red',  # 'blue',
    'binder_1749049094_3': 'red',  # (0.5, 0.2, 0.8),  # RGB tuple (purple-ish)
    'binder_1749049094_4': 'cyan',
    'binder_1749049094_5': 'magenta',
    # Add more if you have more parent IDs, or handle missing ones
}