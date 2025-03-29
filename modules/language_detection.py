import fasttext
import pycountry
import pandas as pd
from tqdm import tqdm
from text_cleaning import clean_crlf, get_last_valid_sentence
from pathlib import Path


# === Project paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data" / "lid.176.bin"

# === Load FastText model (once) ===
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ FastText model not found at {MODEL_PATH}")

model = fasttext.load_model(str(MODEL_PATH))

def get_language_name(lang_code):
    """Try to get the full language name from alpha_2 or alpha_3 code."""
    if not isinstance(lang_code, str):
        return "Unknown"
    lang = pycountry.languages.get(alpha_2=lang_code)
    if lang:
        return lang.name
    lang = pycountry.languages.get(alpha_3=lang_code)
    if lang:
        return lang.name
    return "Unknown"


def detect_language_fasttext(text):
    """
    Detect the language code of a single text using FastText.

    Args:
        text (str): The input text.

    Returns:
        str: Detected language code (e.g., 'en', 'de', 'fr').
    """
    if not isinstance(text, str) or not text.strip():
        return "unknown"

    cleaned = clean_crlf(text)
    cleaned = get_last_valid_sentence(cleaned)
    print(cleaned)
    try:
        prediction = model.predict([cleaned], k=1)
        language_code = prediction[0][0][0].replace("__label__", "")
    except Exception as e:
        print(f"Language detection failed: {e}")
        language_code = "unknown"

    return language_code


def detect_languages_fasttext(df, text_column="ProcessedTextBody", batch_size=10000):
    """Detect languages for a DataFrame column using FastText."""
    if text_column not in df.columns:
        raise ValueError(f"❌ Column '{text_column}' not found in the DataFrame!")

    texts = df[text_column].astype(str).fillna("").map(clean_crlf).map(get_last_valid_sentence).tolist()
    language_codes = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Detecting Languages"):
        batch = texts[i: i + batch_size]
        predictions = model.predict(batch, k=1)
        batch_languages = [pred[0].replace("__label__", "") for pred in predictions[0]]
        language_codes.extend(batch_languages)

    language_names = [get_language_name(code) for code in language_codes]

    df["Language"] = pd.Series(language_codes, index=df.index)
    df["LangName"] = pd.Series(language_names, index=df.index)

    return df


def get_language_distribution(df, lang_column="LangName", threshold=500):
    """
    Generates a top-down list of email counts and percentages by language,
    grouping minor languages under 'Other' (always placed at the bottom).

    Args:
        df (pd.DataFrame): DataFrame containing the language column.
        lang_column (str): Column name with language names.
        threshold (int): Minimum count required to list separately; otherwise, grouped as 'Other'.

    Returns:
        pd.DataFrame: Summary table with language, count, and percentage formatted as '99.99%'.
    """
    # Count occurrences of each language
    language_counts = df[lang_column].value_counts()

    # Calculate total emails
    total_emails = language_counts.sum()

    # Calculate percentage
    language_percentages = (language_counts / total_emails) * 100

    # Create DataFrame
    summary_df = pd.DataFrame({
        "Language": language_counts.index,
        "EmailCount": language_counts.values,
        "Percentage": language_percentages
    })

    # Identify languages below threshold
    mask = summary_df["EmailCount"] < threshold

    # Aggregate 'Other' category
    other_count = summary_df.loc[mask, "EmailCount"].sum()
    other_percentage = summary_df.loc[mask, "Percentage"].sum()

    # Filter out minor languages
    summary_df = summary_df.loc[~mask]

    # Sort in descending order (excluding 'Other')
    summary_df = summary_df.sort_values(by="EmailCount", ascending=False)

    # Append 'Other' row at the bottom if applicable
    if other_count > 0:
        other_row = pd.DataFrame([{
            "Language": "Other",
            "EmailCount": other_count,
            "Percentage": other_percentage
        }])
        summary_df = pd.concat([summary_df, other_row], ignore_index=True)

    # Format percentage column as '99.99%'
    summary_df["Percentage"] = summary_df["Percentage"].apply(lambda x: f"{x:.2f}%")

    return summary_df
