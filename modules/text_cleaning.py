import re
import pandas as pd

# Precompiled universal patterns
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTTP_BRACKET_PATTERN = re.compile(r"\[https?://[^]]+]", re.IGNORECASE)
EMPTY_LINES_PATTERN = re.compile(r"(?m)^[ \t]+$\n?")
NON_PRINTABLE_REGEX = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F\u200b\u200c\u200d\u2060\ufeff]')

# Common header keywords (multi-language)
HEADER_KEYWORDS = {
    "from": ["van", "from", "de", "von", "da", "kimden", "差出人", "发件人"],
    "sent": ["verzonden", "sent", "envoyé", "gesendet", "enviado", "inviato", "gönderildi", "送信日時", "发送时间"],
    "to": ["aan", "to", "à", "an", "para", "a", "宛先", "收件人", "kime"],
    "subject": ["onderwerp", "subject", "objet", "betreff", "asunto", "oggetto", "assunto", "件名", "主题", "konu"],
    "urgency": ["urgentie", "importance", "wichtigkeit", "importancia", "importanza", "importância", "重要度",
                "紧急程度", "önem derecesi"]
}

# Unified reply/forward header detection
OUTLOOK_HEADER_PATTERN = re.compile(
    r"(?is)^.*?(?:from|van|de|von|kimden|差出人).*?\n"
    r"(?:sent|verzonden|envoyé|enviado|gesendet|gönderildi|发送时间).*?\n"
    r"(?:urgentie|importance|wichtigkeit|önem derecesi).*?\n?"
    r"(?:to|aan|para|宛先|收件人|kime).*?\n"
    r"(?:subject|onderwerp|betreff|件名|konu).*?\n"
)

GENERIC_REPLY_PATTERN = re.compile(
    r"(?is)^.*?-{2,}\s*(original message|forwarded message|oorspronkelijk bericht|mensaje original|"
    r"message d'origine|nachricht weitergeleitet|mensagem encaminhada|転送されたメッセージ|转发邮件|iletilmiş mesaj)\s*-+"
)

GMAIL_APPLE_HEADER_PATTERN = re.compile(
    r"(?is)^.*?(on|op|le|am|el|於|于|tarihinde)\s.*?(wrote|geschreven|écrit|escribió|scrisse|escreveu|yazdı|"
    r"さんが書きました|写道):\s*"
)

# Discardable disclaimers or automated banners
DISCARD_PATTERNS = [
    re.compile(r"ATTENTION: This is an external email.*?safe\.", re.IGNORECASE | re.DOTALL),
    re.compile(r"Please consider the environment.*?email\.", re.IGNORECASE),
    re.compile(r"This message and any attachments are intended only for the addressee.*?", re.IGNORECASE | re.DOTALL),
    re.compile(r"https?://checkpoint\.url-protection\.com/\S+", re.IGNORECASE),
    re.compile(r"The attached message was sent.*?Sent Items folder after the upgrade\.", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"The information contained in this communication is confidential and may be legally privileged\. "
        r"It is intended solely for the use of the individual or the entity to whom it is addressed and the others authorized to receive it\. "
        r"If you are not the intended recipient, you are hereby notified that any disclosure, copying, distribution or taking any action in reliance of the contents of this information is strictly prohibited and that the content of the message should be deleted\. "
        r"Any e-mail messages from Agristo are given in good faith but shall not be binding nor shall they construe any obligation\.",
        re.IGNORECASE | re.DOTALL
    )
]

# Sensitive info patterns
EMAIL = r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"
PHONE = r"(phone|fax)?[\s:=\-]*\+?\(?\d{1,4}\)?[\d\s\-/.]{5,}"
DOMAIN = r"(https?://)?(www\.)?[a-z0-9\-]+\.(com|net|org|edu|gov|[a-z]{2,})(/[^\s]*)?"
TITLE = r"(manager|director|engineer|officer|coordinator|specialist|lead|president|consultant|analyst|developer|ceo|cto|cfo)"
ADDRESS = r"\d{1,5}[ ,\-]?\s*(straat|street|road|avenue|drive|laan|weg|plaza|parc|port|boulevard)\b.*"

# Combined sensitive pattern for full-line matching
SENSITIVE_LINE_PATTERN = re.compile(
    rf"(?i)^\s*({EMAIL}|{PHONE}|{DOMAIN}|{TITLE}|{ADDRESS})\s*$",
    re.IGNORECASE | re.MULTILINE
)

# Token replacements (optional)
PATTERN_REPLACEMENTS = {
    re.compile(rf"{EMAIL}", re.IGNORECASE): "[EMAIL]",
    re.compile(rf"{PHONE}", re.IGNORECASE): "[PHONE]",
    re.compile(rf"{DOMAIN}", re.IGNORECASE): "[DOMAIN]",
    re.compile(rf"{TITLE}", re.IGNORECASE): "[TITLE]",
}

# Matches the full-line thread ID (only content on the line)
THREAD_ID_LINE_PATTERN = re.compile(r"^\s*thread::[a-zA-Z0-9]+(?:::)?\s*$", re.IGNORECASE)

# Matches [ thread::...:: ] with optional spaces
BRACKETED_THREAD_PATTERN = re.compile(r"\[\s*thread::[a-zA-Z0-9]+(?:::)?\s*\]", re.IGNORECASE)

CID_IMAGE_PATTERN = re.compile(r"\[cid:[^]]+]", re.IGNORECASE)


# ----------------------------
# Cleaning Functions
# ----------------------------

def remove_cid_references(text):
    return CID_IMAGE_PATTERN.sub("", text) if isinstance(text, str) else text


def clean_thread_ids(text):
    if not isinstance(text, str):
        return text

    subject_keywords = HEADER_KEYWORDS.get("subject", [])
    cleaned_lines = []

    for line in text.splitlines():
        # Remove [ thread::...:: ] with brackets
        line = BRACKETED_THREAD_PATTERN.sub("", line)

        # Remove any remaining thread::...:: patterns (including special characters, safely)
        line = THREAD_ID_LINE_PATTERN.sub("", line) if THREAD_ID_LINE_PATTERN.fullmatch(line.strip()) else \
            re.sub(r"thread::[^:\]\s]+(?:::)?", "", line, flags=re.IGNORECASE)

        # Clean thread ID from subject-like lines
        lower = line.strip().lower()
        if any(lower.startswith(k + ":") for k in subject_keywords):
            line = BRACKETED_THREAD_PATTERN.sub("", line).strip()

        cleaned_lines.append(line.strip())

    return "\n".join(cleaned_lines)


def remove_http_bracket_lines(text):
    return HTTP_BRACKET_PATTERN.sub("", text) if isinstance(text, str) else text


def remove_non_printable(text):
    return NON_PRINTABLE_REGEX.sub('', text) if isinstance(text, str) else text


def reduce_multiple_spaces(text):
    return re.sub(r"[ \t]+", " ", text)


def remove_empty_lines(text):
    if not isinstance(text, str):
        return text
    # Remove lines with only whitespace
    text = EMPTY_LINES_PATTERN.sub("", text)
    # Then collapse multiple empty lines to one
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()


def remove_html(text):
    return HTML_TAG_PATTERN.sub(" ", text)


def apply_pattern_substitutions(text):
    for pattern, token in PATTERN_REPLACEMENTS.items():
        text = pattern.sub("", text)
    return text


def apply_discard_patterns(text):
    for pattern in DISCARD_PATTERNS:
        text = pattern.sub("", text)
    return text


def strip_known_reply_headers(text):
    for pattern in [OUTLOOK_HEADER_PATTERN, GENERIC_REPLY_PATTERN, GMAIL_APPLE_HEADER_PATTERN]:
        new = pattern.sub("", text)
        if new != text:
            return new.strip()
    return text


def strip_flexible_reply_header(text, min_matches=3, max_scan_lines=30):
    if not isinstance(text, str):
        return text
    lines = text.splitlines()
    match_lines = []
    lower_lines = [line.lower() for line in lines[-max_scan_lines:]]
    offset = len(lines) - len(lower_lines)

    for i, line in enumerate(lower_lines):
        for key, keywords in HEADER_KEYWORDS.items():
            if any(line.strip().startswith(k + ":") for k in keywords):
                match_lines.append((offset + i, key))
                break

    if len({k for _, k in match_lines}) >= min_matches:
        # Find the first occurrence of a subject-like line
        for idx, key in match_lines:
            if key == "subject":
                return "\n".join(lines[idx:]).strip()  # Preserve from "Subject:" line onward
        # fallback to old behavior if no subject found
        return "\n".join(lines[max(i for i, _ in match_lines) + 1:]).strip()

    return text


def remove_sensitive_lines(text):
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append(line)
        elif not re.search(SENSITIVE_LINE_PATTERN, stripped):
            lines.append(line)
    return "\n".join(lines)


def get_last_valid_sentence(text):
    if not isinstance(text, str):
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    for sentence in reversed(sentences):
        if len(sentence.split()) >= 8 and re.search(r'[.!?]$', sentence.strip()):
            return sentence.strip()
    return " ".join(text.strip().split()[:20])


def clean_crlf(text):
    return text.replace("\n", " ").replace("\r", " ").strip() if isinstance(text, str) else ""


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = strip_flexible_reply_header(text)
    text = remove_empty_lines(text)
    text = reduce_multiple_spaces(text)
    text = remove_non_printable(text)
    text = remove_http_bracket_lines(text)
    text = clean_thread_ids(text)
    text = remove_cid_references(text)
    text = strip_known_reply_headers(text)
    text = remove_html(text)
    text = apply_discard_patterns(text)
    # text = apply_pattern_substitutions(text)
    text = remove_sensitive_lines(text)
    return text.strip()


def filter_short_texts(df: pd.DataFrame, column: str = "ProcessedTextBody", threshold: int = 50) -> pd.DataFrame:
    """
    Remove rows where the character count of the specified column falls below the given threshold.

    :param df: Input DataFrame.
    :param column: Column name to check character length.
    :param threshold: Minimum number of characters required to keep the row.
    :return: Filtered DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    return df[df[column].str.len() >= threshold].reset_index(drop=True)
