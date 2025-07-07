# scripts/plot_al_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import logging
import math  # For math.ceil in embedding plot

# Configure basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _load_and_filter_data(al_state_csv_path: str):
    """
    Helper function to load the AL state CSV and perform common filtering.
    """
    if not os.path.exists(al_state_csv_path):
        logger.error(f"AL state CSV file not found at: {al_state_csv_path}")
        raise FileNotFoundError(f"CSV not found: {al_state_csv_path}")

    try:
        df = pd.read_csv(
            al_state_csv_path,
            # Specify dtypes for robustness. Include all relevant columns.
            dtype={
                'id': str, 'parent_id': str, 'original_parent_id': str,
                'embedding_path': str,  'iteration': pd.Int64Dtype(),
                'iptm': float, 'ptm': float, 'ranking_score': float,
                'af3_status': str, 'binder_sequence': str, 'target_id': str,
                # Add other columns if they are relevant for plotting/analysis later
                'mutation_type': str, 'original_aa_pos_mut': str
            },
            keep_default_na=True,  # Allow default NaN handling
            na_values=['']  # Treat empty strings as NaN
        )
        logger.info(f"Loaded {len(df)} entries from {al_state_csv_path}")

        # Post-process columns that are expected to be actual Python objects (lists/tuples)
        # Apply ast.literal_eval if the column is present and contains non-NaN strings
        # ONLY IF you have lists/tuples stringified in these columns
        # import ast
        # if 'mutation_history' in df.columns:
        #    df['mutation_history'] = df['mutation_history'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) and x.strip() else [])
        # if 'mutations_applied' in df.columns:
        #    df['mutations_applied'] = df['mutations_applied'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) and x.strip() else [])

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
        ref_line_value: float = 1.0  # Reference line applies to both plots
):
    """
    Plots mean and max IPTM/PTM metrics against iterations using line plots.
    """
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
    plt.axhline(y=ref_line_value, color='black', linestyle=':', linewidth=1.5)
    plt.title('IPTM Score Trajectory')
    plt.xlabel('Active Learning Iteration')
    plt.ylabel('IPTM Score')
    plt.ylim(0, 1)
    plt.xticks(metrics_summary['iteration'])
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sns.lineplot(data=metrics_summary, x='iteration', y='mean_ptm', marker='o', label='Mean PTM', color='lightcoral')
    sns.lineplot(data=metrics_summary, x='iteration', y='max_ptm', marker='x', label='Max PTM', color='darkred')
    plt.axhline(y=ref_line_value, color='black', linestyle=':', linewidth=1.5)
    plt.title('PTM Score Trajectory')
    plt.xlabel('Active Learning Iteration')
    plt.ylabel('PTM Score')
    plt.ylim(0, 1)
    plt.xticks(metrics_summary['iteration'])
    plt.legend()
    plt.grid(True)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(output_plot_path, dpi=600)
    logger.info(f"Line plot saved to {output_plot_path}")
    plt.close()


def plot_al_violin_trajectory(
        df: pd.DataFrame,
        output_plot_path: str,
        metric_column: str = 'iptm',
        title: str = "Active Learning Trajectory of AF3 Metrics",
        y_label: str = "Score",
        reference_line_value: float = 1.0,
        color_split_iteration: int = 3
):
    """
    Plots violin plots with overlaid data points for a given metric over iterations,
    mimicking the provided image.
    """
    df_evaluated = df[
        (df['af3_status'] == 'success') &
        (df[metric_column].notna())
        ].copy()

    if df_evaluated.empty:
        logger.warning(
            f"No successful AF3 evaluations with valid {metric_column} found for violin plot. Plot not generated.")
        return

    df_evaluated['Phase'] = df_evaluated['iteration'].apply(
        lambda x: 'Early' if x <= color_split_iteration else 'Late'
    )

    custom_palette = {'Early': 'tab:orange', 'Late': 'tab:blue'}

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))

    sns.violinplot(
        x='iteration',
        y=metric_column,
        hue='Phase',
        data=df_evaluated,
        palette=custom_palette,
        inner='quartile',
        linewidth=1.5,
        edgecolor='blue',
        cut=0
    )

    sns.stripplot(
        x='iteration',
        y=metric_column,
        hue='Phase',
        data=df_evaluated,
        palette={'Early': 'white', 'Late': 'white'},
        edgecolor='black',
        linewidth=1.0,
        jitter=0.2,
        size=7,
        dodge=True,
        ax=plt.gca(),
        zorder=1
    )

    plt.axhline(y=reference_line_value, color='black', linestyle=':', linewidth=1.5)

    plt.title(title, fontsize=16)
    plt.xlabel('Active Learning Iteration', fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    plt.ylim(0.0, 1.0)  # Assume 0-2 range for scores like IPTM/PTM. Adjust if needed.

    # Ensure x-axis ticks show only integer iteration numbers present in data
    # Get unique iterations from the filtered DataFrame, then cast to int for ticks
    if not df_evaluated['iteration'].empty:
        plt.xticks(df_evaluated['iteration'].unique().astype(int))

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles=handles[:len(custom_palette)], labels=labels[:len(custom_palette)], title="Phase")

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=600)
    logger.info(f"Violin plot saved to {output_plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate various plots for Active Learning trajectory.")
    parser.add_argument('--input_csv', type=str, default='data/outputs/al_state.csv',
                        help='Path to the al_state.csv file.')
    parser.add_argument('--output_dir', type=str, default='plots/',
                        help='Directory to save output plots.')
    parser.add_argument('--plot_type', type=str, choices=['line', 'violin', 'all'], default='all',
                        help='Type of plot to generate: "line" (mean/max trajectory), "violin" (distribution per iteration), or "all".')

    # Arguments specific to plotting styles
    parser.add_argument('--metric', type=str, default='iptm',
                        help='Which metric to plot for violin plot (e.g., "iptm", "ptm", "ranking_score").')
    parser.add_argument('--title', type=str, default='BindSculpt Active Learning Trajectory',
                        help='Main title for the plots.')
    parser.add_argument('--y_label', type=str, default='Score',
                        help='Label for the Y-axis (primarily for violin plot).')
    parser.add_argument('--ref_line_value', type=float, default=1.5,
                        help='Value for the horizontal dotted reference line in plots.')
    parser.add_argument('--color_split_iter', type=int, default=3,
                        help='Iteration number (inclusive) for color split in violin plot.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        df_full = _load_and_filter_data(args.input_csv)
        if df_full.empty:
            logger.error("No data available after loading/filtering. Exiting plotting script.")
            return

        if args.plot_type == 'line' or args.plot_type == 'all':
            output_line_plot_path = os.path.join(args.output_dir, f"{args.metric}_line_trajectory.svg")
            plot_al_line_trajectory(
                df=df_full,
                output_plot_path=output_line_plot_path,
                title=args.title,
                ref_line_value=args.ref_line_value
            )

        if args.plot_type == 'violin' or args.plot_type == 'all':
            # For violin plot, we need to pass the y_label specific to the metric chosen
            violin_y_label = f"{args.metric.upper()} Score" if args.y_label == 'Score' else args.y_label
            output_violin_plot_path = os.path.join(args.output_dir, f"{args.metric}_violin_trajectory.svg")
            plot_al_violin_trajectory(
                df=df_full,
                output_plot_path=output_violin_plot_path,
                metric_column=args.metric,
                title=args.title,
                y_label=violin_y_label,
                reference_line_value=args.ref_line_value,
                color_split_iteration=args.color_split_iter
            )

    except FileNotFoundError:
        # Already logged in _load_and_filter_data
        pass
    except Exception as e:
        logger.critical(f"A critical error occurred in the plotting script: {e}", exc_info=True)


if __name__ == "__main__":
    main()