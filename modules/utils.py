import chardet
from pandarallel import pandarallel
import pandas as pd
import numpy as np

from langdetect import detect, DetectorFactory
from text_cleaning import clean_text

# Ensure consistent language detection results
DetectorFactory.seed = 0

# Ensure Pandas displays all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # Set a wider display
pd.set_option('display.expand_frame_repr', False)  # Prevent column wrapping


def show_unique_values(df, col):
    """
    Prints the unique values and counts of a specified column, including NaN values.

    Parameters:
    df (pd.DataFrame): The DataFrame.
    col (str): The column name to analyze.
    """
    if col not in df.columns:
        print(f"⚠️ Column '{col}' not found in DataFrame.")
        return

    print("\n" + "=" * 60)
    print(f"📊 Unique Values in Column: {col}")
    print("=" * 60)

    # Value counts (including NaN values)
    column_counts = df[col].value_counts(dropna=False)
    print(column_counts)

    # Count of NaN values
    nan_count = df[col].isna().sum()
    print(f"\n⚠️ Number of NaN values in '{col}': {nan_count}")


def remove_duplicate_columns(df):
    cleaned_df = df.loc[:, ~df.columns.duplicated()]
    return cleaned_df


def remove_all_nan_columns(df):
    """Removes columns from a DataFrame where all values are NaN efficiently."""
    before_cols = set(df.columns)  # Save columns before removal

    pd.set_option("future.no_silent_downcasting", True)
    # Assign `.replace()` result instead of using `inplace=True`
    df = df.replace(["", "nan"], np.nan).infer_objects(copy=False)

    # Drop columns where all values are NaN
    df = df.dropna(axis=1, how='all')

    # De-fragment the DataFrame by making a full copy
    df = df.copy()  # Fixes PerformanceWarning

    after_cols = set(df.columns)  # Save columns after removal

    # Find which columns were removed
    removed_cols = before_cols - after_cols
    # if removed_cols:
    #     print(f"✅ Removed {len(removed_cols)} all-NaN columns: {removed_cols}")
    # else:
    #     print("⚠️ No all-NaN columns were removed.")

    return df  # Return the updated DataFrame


def find_all_nan_columns(df):
    """Returns columns where all values are NaN."""
    nan_columns = df.columns[df.isna().all()].tolist()
    return nan_columns


def detect_language_fast(texts):
    """Performs batch language detection for efficiency."""
    languages = []
    for text in texts:
        try:
            languages.append(detect(text))
        except:
            languages.append("unknown")
    return languages


def detect_file_encoding(file_path, sample_size=100000):
    """
    Detects the encoding of a given file by reading a sample of its content.

    :param file_path: Path to the file.
    :param sample_size: Number of bytes to read for detection (default: 100,000 bytes).
    :return: Detected encoding and confidence level.
    """
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read(sample_size))

    encoding = result.get("encoding", "utf-8")
    confidence = result.get("confidence", 0)

    print(f"Detected Encoding: {encoding} (Confidence: {confidence})")
    return encoding, confidence


def read_csv_file(file_path):
    """Reads a CSV file with UTF-8 encoding and returns a DataFrame."""
    try:
        return pd.read_csv(file_path, sep=';', dtype=str, encoding='utf-8', low_memory=False)
    except FileNotFoundError as fnf_error:
        print(f"❌ File not found: {fnf_error}")
        return None
    except pd.errors.ParserError as parse_error:
        print(f"❌ Error parsing CSV file: {parse_error}")
        return None


def merge_datasets(emails_df, metadata_df, left_on="ParentId", right_on="AGR_Id", how="inner"):
    """Merges emails DataFrame with metadata DataFrame on ParentId -> AGR_Id."""
    return emails_df.merge(metadata_df, left_on=left_on, right_on=right_on, how=how)
