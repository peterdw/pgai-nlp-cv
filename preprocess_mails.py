from language_detection import detect_languages_fasttext
from utils import read_csv_file, \
    find_all_nan_columns, remove_all_nan_columns, remove_duplicate_columns, clean_both_versions
from tqdm import tqdm


def preprocess(df_emails):
    """Cleans emails and detects language using FastText, avoiding redundant operations."""
    if "TextBody" not in df_emails.columns:
        print("⚠️ 'TextBody' column not found.")
        return None

    # Convert to string and apply combined cleaning and tagging in one go
    tqdm.pandas(desc="Cleaning & Tagging Emails")
    df_emails["TextBody"] = df_emails["TextBody"].astype(str)
    df_emails[["ProcessedTextBody"]] = df_emails.progress_apply(clean_both_versions, axis=1)

    # FastText language detection
    df_emails = detect_languages_fasttext(df_emails, text_column="ProcessedTextBody")

    return df_emails


def explore_dataframe(datafr):
    """
    Performs comprehensive exploratory data analysis on a DataFrame and prints formatted results.

    Parameters:
    df (pd.DataFrame): The DataFrame to be analyzed.
    """

    print("\n" + "=" * 60)
    print(f"📊 DATAFRAME EXPLORATION REPORT")
    print("=" * 60)

    # 1️⃣ Basic Info
    print("\n🔹 Shape of DataFrame (rows, columns):", datafr.shape)

    # 2️⃣ Column Names & Data Types (No Alignment Issues)
    print("\n🔹 Column Names and Data Types:")
    for col, dtype in datafr.dtypes.items():
        print(f"   {col}: {dtype}")

    # 3️⃣ Missing Values
    missing_values = datafr.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if missing_values.empty:
        print("\n✅ No Missing Values")
    else:
        print("\n⚠️ Missing Values:")
        for col, count in missing_values.items():
            print(f"   {col}: {count} missing")

    # 4️⃣ Duplicates
    duplicate_count = datafr.duplicated().sum()
    print(f"\n🔄 Duplicate Rows: {duplicate_count}")

    # 5️⃣ Summary Statistics
    print("\n📈 Summary Statistics (Numerical Columns):\n", datafr.describe())
    print("\n📊 Summary Statistics (Categorical Columns):\n", datafr.describe(include=[object]))

    # 6️⃣ Unique Values per Column
    print("\n🔢 Unique Values Per Column:")
    for col in datafr.columns:
        unique_vals = datafr[col].nunique()
        print(f"   {col}: {unique_vals} unique values")

    # 7️⃣ Distinct Values for Columns with <30 Unique Values
    print("\n🔍 Distinct Values in Columns with <30 Unique Values:")
    for col in datafr.columns:
        unique_vals = datafr[col].nunique()
        if unique_vals < 30:  # Only show details if there are less than 30 unique values
            print(f"\n▶ {col} ({unique_vals} unique values):")
            print(datafr[col].dropna().unique().tolist())  # Print as a Python list for better readability

    # # 8️⃣ Top 5 Frequent Values Per Column
    # print("\n🔝 Top 5 Frequent Values Per Column:")
    # for col in df.columns:
    #     print(f"\n▶ {col}:")
    #     print(df[col].value_counts(dropna=False).head(5))

    # # 9️⃣ Correlation Matrix (Numerical Data)
    # if df.select_dtypes(include=['number']).shape[1] > 1:
    #     print("\n📊 Correlation Matrix (Numerical Data):\n", df.corr())

    # # 🔟 Memory Usage
    # print("\n💾 Memory Usage:")
    # print(df.memory_usage(deep=True))

    # # 🏁 First & Last Rows
    # print("\n🧐 First 5 Rows:\n", df.head())
    # print("\n🔍 Last 5 Rows:\n", df.tail())

    print("\n" + "=" * 60)
    print("✅ DATA EXPLORATION COMPLETED")
    print("=" * 60)


def main():
    idx = None
    limit = None
    """Loads and processes emails from CSV files with optimized merging."""
    emails_file = "data/salesforce_case.csv"
    metadata_file = "data/metadata.csv"

    df_emails = read_csv_file(emails_file)
    df_meta = read_csv_file(metadata_file)

    if df_emails is None or df_meta is None:
        print("\u274C Missing dataset files.")
        return None

    # remove all columns without values (NaN)
    df_emails = remove_all_nan_columns(df_emails)
    df_meta = remove_all_nan_columns(df_meta)

    df_emails = remove_duplicate_columns(df_emails)
    df_meta = remove_duplicate_columns(df_meta)

    df_meta = df_meta.rename(columns=lambda col: f"META_{col}")

    # Merge only required columns for efficiency
    df_merged = df_emails.merge(df_meta, left_on="ParentId", right_on="META_Id", how="inner")
    # Add Complaint column
    df_merged["Complaint"] = (~(
            df_merged["AGR_Type_of_Complaint__c"].isna() |
            (df_merged["AGR_Type_of_Complaint__c"].str.strip().str.lower() == "not applicable")
    )).astype("int64")
    df_merged = remove_duplicate_columns(df_merged)
    df_merged = remove_all_nan_columns(df_merged)

    nan_columns = find_all_nan_columns(df_emails)
    if nan_columns:
        print("Columns with all NaN values in df_emails:", nan_columns)

    nan_columns = find_all_nan_columns(df_meta)
    if nan_columns:
        print("Columns with all NaN values in df_meta:", nan_columns)

    nan_columns = find_all_nan_columns(df_merged)
    if nan_columns:
        print("Columns with all NaN values in df_merged:", nan_columns)

    # print(df_emails.columns.tolist())
    # print(df_meta.columns.tolist())
    # print(df_merged.columns.tolist())

    if idx is not None:
        if idx < 0 or idx >= len(df_merged):
            print("\u26A0 Index out of range.")
            return None
        df_merged = df_merged.iloc[[idx]]
    elif limit is not None:
        df_merged = df_merged.head(limit)

    df_processed = preprocess(df_merged)
    return df_processed


if __name__ == "__main__":
    df = main()
    # Save DataFrame as Feather
    df.to_feather("data/df_emails_cleaned.feather")
