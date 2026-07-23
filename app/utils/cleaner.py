# app/utils/cleanner.py
"""
function to clean raw web results from tavily search:
    Handle None values
    Strip whitespace
    Normalize whitespace/newlines
    Remove common website boilerplate
    Remove repeated headings
    Drop very short documents
    Drop duplicate documents
    Compute metadata (length, words)
    Return cleaned documents
"""

import pandas as pd
import re

# Common boilerplate phrases found in Tavily extracted pages
BOILERPLATE_PATTERNS = [
    r"Join over \d[\d,]* subscribers.*?(?=How|Examples|Frequently|$)",
    r"Thank you!\s*You are subscribed\.",
    r"Share\s*Link copied",
    r"Take the next step.*",
    r"Resources.*",
    r"Related solutions.*",
    r"Read the report",
    r"Watch now",
    r"Listen now",
    r"Get the guide",
    r"Explore .*",
]


def clean(raw_texts: list[str]) -> list[dict]:
    df = pd.DataFrame({"raw": raw_texts})

    # replace none with empty string
    df["raw"] = df["raw"].fillna("")

    # remove trailing/leading whitespace
    df["clean"] = df["raw"].str.strip()

    # normalize newlines and whitespaces
    df["clean"] = df["clean"].apply(lambda x: re.sub(r"\s+", " ", x))

    # remove boilerplate patters
    for pattern in BOILERPLATE_PATTERNS:
        df["clean"] = df["clean"].apply(
            lambda x: re.sub(pattern, "flags=re.IGNORECASE | re.DOTALL", x)
        )

    # remove duplicate consecutive headings
    df["clean"] = df["clean"].apply(lambda x: re.sub(r"\b(.{5,80})\b\s+\1\b", r"\1", x))

    # remove whitespace again
    df["clean"] = df["clean"].str.strip()

    # remove very short entries
    df = df[df["clean"].str.len() > 150]

    # remove duplicates
    df = df.drop_duplicates(subset="clean")

    # return list of {'raw':..., 'clean':...}
    return df.to_dict("records")
