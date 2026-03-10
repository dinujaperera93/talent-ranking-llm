import re
from config import CONNECTIONS_MAX

def clean_job_title(title):
    title = str(title).lower().strip()
    title = re.sub(r"[^\w\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip()

def parse_connections(value):
    value = str(value).strip()
    if value.endswith("+"):
        return CONNECTIONS_MAX
    try:
        return int(value)
    except ValueError:
        return 0

def preprocess(df):
    df = df.copy()
    df["job_title_clean"] = df["job_title"].apply(clean_job_title)
    df["connections_raw"] = df["connection"].apply(parse_connections)
    df["connections_norm"] = (df["connections_raw"] / CONNECTIONS_MAX).clip(0, 1)
    return df
