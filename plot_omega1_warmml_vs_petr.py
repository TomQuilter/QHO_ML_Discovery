from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# User settings: edit these values to change the plot quickly.
# ============================================================

# Data file to read.
# This script now supports BOTH Excel files (.xlsx, .xls) and CSV files (.csv).
# Example Excel: "Omega_1_WarmML_vs_Pet.xlsx"
# Example CSV:   "Omega_1_WarmML_vs_Pet.csv"
# Default is set to your CSV file.
DATA_PATH = "Omega_1_WarmML_vs_Pet.csv"

# Sheet name is only used for Excel files.
# Leave as None to use the first sheet in the workbook.
SHEET_NAME = None

# Column names to plot against each other.
X_COLUMN = "lambda" 
Y1_COLUMN = "Pet_Real"
Y2_COLUMN = "ML_Real"
  
# Optional filter by seed.
# Example: SEED_FILTER = 45
# Leave as None to keep all rows.
SEED_FILTER = None

# Optional lambda range.
# Leave as None for full range.
LAMBDA_MIN = None
# Example: set to 1.0 to plot only the first few rows (lambda < 1).
LAMBDA_MAX = 1

# Plot appearance.
FIGSIZE = (10, 6)
TITLE = r"Perturbation Real vs ML Real against $\lambda$"
X_LABEL = r"$\lambda$"
Y_LABEL = "Value"
Y1_LABEL = "Pet_Real"
Y2_LABEL = "ML_Real"
Y1_COLOR = "navy"  
Y2_COLOR = "crimson"
Y1_MARKER = "o" 
Y2_MARKER = "s"
Y1_LINESTYLE = "-"
Y2_LINESTYLE = "--"
GRID_ALPHA = 0.3

# Set to True if you want a PNG written to disk automatically.
# Set to False for inline/display-only plotting.
SAVE_FIGURE = False
OUTPUT_PATH = "Omega_1_WarmML_vs_Pet_plot.png"
OUTPUT_DPI = 300

# Set this to True to display the plot only (no saved file).
SHOW_FIGURE = True


def load_data(data_path: str, sheet_name=None) -> pd.DataFrame:
    """
    Read either an Excel sheet or a CSV file.

    - If the file is .csv, the script reads it directly.
    - If the file is Excel, the script uses the chosen sheet.
    """
    suffix = Path(data_path).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(data_path)

    if suffix in {".xlsx", ".xls"}:
        if sheet_name is None:
            excel_file = pd.ExcelFile(data_path)
            first_sheet = excel_file.sheet_names[0]
            return pd.read_excel(data_path, sheet_name=first_sheet)

        return pd.read_excel(data_path, sheet_name=sheet_name)

    raise ValueError(
        "Unsupported file type. Please use a .csv, .xlsx, or .xls file."
    )


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and sort the data so lambda is treated as numeric, not categorical.

    Steps:
    1. Keep only the columns needed for the plot.
    2. Convert lambda and y-columns to numeric.
    3. Drop rows with missing values in those columns.
    4. Optionally filter by seed.
    5. Optionally filter by lambda range.
    6. Sort by lambda so lines are drawn in the correct numerical order.
    """
    required_columns = [X_COLUMN, Y1_COLUMN, Y2_COLUMN]

    if SEED_FILTER is not None:
        if "seed" not in df.columns:
            raise KeyError("SEED_FILTER was set, but the sheet has no 'seed' column.")
        df = df[df["seed"] == SEED_FILTER]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    plot_df = df[required_columns].copy()

    # Force columns to numeric so lambda is plotted on a true number axis.
    plot_df[X_COLUMN] = pd.to_numeric(plot_df[X_COLUMN], errors="coerce")
    plot_df[Y1_COLUMN] = pd.to_numeric(plot_df[Y1_COLUMN], errors="coerce")
    plot_df[Y2_COLUMN] = pd.to_numeric(plot_df[Y2_COLUMN], errors="coerce")

    plot_df = plot_df.dropna(subset=required_columns)

    if LAMBDA_MIN is not None:
        plot_df = plot_df[plot_df[X_COLUMN] >= LAMBDA_MIN]

    if LAMBDA_MAX is not None:
        plot_df = plot_df[plot_df[X_COLUMN] <= LAMBDA_MAX]

    plot_df = plot_df.sort_values(by=X_COLUMN).reset_index(drop=True)

    if plot_df.empty:
        raise ValueError("No rows remain after filtering. Check seed/range settings.") 

    return plot_df


def make_plot(plot_df: pd.DataFrame) -> None:
    """
    Draw a clear comparison plot of Pet_Real and ML_Real versus lambda.

    To change colours, markers, labels, or line styles, edit the settings
    near the top of this file rather than changing this function.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(
        plot_df[X_COLUMN],
        plot_df[Y1_COLUMN],
        color=Y1_COLOR,
        marker=Y1_MARKER,
        linestyle=Y1_LINESTYLE,
        linewidth=2,
        markersize=6,
        label=Y1_LABEL,
    )

    ax.plot(
        plot_df[X_COLUMN],
        plot_df[Y2_COLUMN],
        color=Y2_COLOR,
        marker=Y2_MARKER,
        linestyle=Y2_LINESTYLE,
        linewidth=2,
        markersize=6,
        label=Y2_LABEL,
    )

    ax.set_title(TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()

    # Tight layout avoids clipped axis labels and titles.
    fig.tight_layout()

    if SAVE_FIGURE:
        fig.savefig(OUTPUT_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")
        print(f"Saved plot to: {OUTPUT_PATH}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    df = load_data(DATA_PATH, SHEET_NAME)
    plot_df = prepare_data(df)

    print("Using columns:")
    print(f"  x  = {X_COLUMN}")
    print(f"  y1 = {Y1_COLUMN}")
    print(f"  y2 = {Y2_COLUMN}")
    print(f"Rows plotted: {len(plot_df)}")
    print(f"Lambda range: {plot_df[X_COLUMN].min()} to {plot_df[X_COLUMN].max()}")

    make_plot(plot_df)


if __name__ == "__main__":
    main()
