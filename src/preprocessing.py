import re
import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import CONNECTIONS_MAX

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

def clean_job_title(title):
    # 1. Lowercase
    title = str(title).lower().strip()
    # 2. Remove punctuation
    title = re.sub(r"[^\w\s]", " ", title)
    tokens = title.split()
    # 3. Remove stop words
    tokens = [t for t in tokens if t not in _STOP_WORDS]
    # 4. Keep alphabetic tokens only
    tokens = [t for t in tokens if t.isalpha()]
    # 5. Lemmatization
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    # 6. Remove short tokens (len <= 2)
    tokens = [t for t in tokens if len(t) > 2]
    return " ".join(tokens)

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
