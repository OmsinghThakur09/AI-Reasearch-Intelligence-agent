# app/utils/cleaner.py
"""
function to clean raw web results from tavily search:
    Handle None values
    Strip whitespace
    Normalize whitespace/newlines
    Remove common website boilerplate (line-anchored, not open-ended)
    Remove repeated headings
    Drop very short documents (by real word count)
    Drop duplicate / near-duplicate documents
    Return cleaned documents
"""

import pandas as pd
import re

# Boilerplate patterns are matched against WHOLE LINES ONLY (via fullmatch),
# applied BEFORE whitespace is collapsed. This prevents a single nav/footer
# word like "Resources" or "Explore" from wiping out everything after it in
# the document, which is what an open-ended ".*" pattern applied to the
# whole blob would do.
BOILERPLATE_LINE_PATTERNS = [
    r"Join over [\d,]+ subscribers.*",
    r"Thank you!\s*You are subscribed\.?",
    r"Share\s*Link copied",
    r"Take the next step.*",
    r"Resources",
    r"Related solutions",
    r"Read the report",
    r"Watch now",
    r"Listen now",
    r"Get the guide",
    r"Explore .*",
    r"Sign up for a free account.*",
    r"Sign in",
    r"Jump to content",
    r"Next steps on \w+.*",
]

_BOILERPLATE_RE = [
    re.compile(p, flags=re.IGNORECASE) for p in BOILERPLATE_LINE_PATTERNS
]

MIN_WORD_COUNT = 150  # real word count now, not character count
NEAR_DUP_PREFIX_WORDS = 40  # words compared when checking near-duplicates


def _is_boilerplate_line(line: str) -> bool:
    "return True if a single line is entirely a boilerplate/nav/footer line"
    stripped = line.strip()
    if not stripped:
        return True
    for pattern in _BOILERPLATE_RE:
        if pattern.fullmatch(stripped):
            return True
    return False


def _strip_boilerplate(text: str) -> str:
    "drop boilerplate lines individually, keep every real content line intact"
    lines = text.split("\n")
    kept = [ln for ln in lines if not _is_boilerplate_line(ln)]
    return "\n".join(kept)


def _word_count(text: str) -> int:
    return len(text.split())


def _near_dup_key(text: str) -> str:
    "cheap near-duplicate signature: normalized first N words"
    words = re.sub(r"\s+", " ", text).strip().lower().split()
    return " ".join(words[:NEAR_DUP_PREFIX_WORDS])


def clean(raw_texts: list[str]) -> list[dict]:
    df = pd.DataFrame({"raw": raw_texts})

    # replace none with empty string
    df["raw"] = df["raw"].fillna("")

    # strip leading/trailing whitespace on the raw blob
    df["clean"] = df["raw"].str.strip()

    # remove boilerplate line-by-line, BEFORE collapsing whitespace,
    # so nav/footer lines can be identified and dropped without touching
    # real content elsewhere in the document
    df["clean"] = df["clean"].apply(_strip_boilerplate)

    # normalize newlines and whitespace (safe to collapse now)
    df["clean"] = df["clean"].apply(lambda x: re.sub(r"\s+", " ", x))

    # remove duplicate consecutive headings/phrases
    df["clean"] = df["clean"].apply(
        lambda x: re.sub(r"\b(.{5,80}?)\b\s+\1\b", r"\1", x, flags=re.IGNORECASE)
    )

    df["clean"] = df["clean"].str.strip()

    # drop very short entries using real word count, not character count
    df["word_count"] = df["clean"].apply(_word_count)

    # --- DIAGNOSTIC: inspect each doc's length + preview before dropping ---
    # for i, row in df.iterrows():
    #     preview = row["clean"][:120].replace("\n", " ")
    #     print(f"[doc {i}] words={row['word_count']:<5} preview: {preview!r}")
    # -------------------------------------------------------------------

    df = df[df["word_count"] > MIN_WORD_COUNT]

    # drop exact duplicates
    df = df.drop_duplicates(subset="clean")

    # drop near-duplicates (e.g. same article fetched twice via two
    # differently-worded search queries, so text isn't byte-identical
    # but opens with the same content)
    df["near_dup_key"] = df["clean"].apply(_near_dup_key)
    df = df.drop_duplicates(subset="near_dup_key")

    df = df.drop(columns=["word_count", "near_dup_key"])

    # return list of {'raw':..., 'clean':...}
    return df.to_dict("records")


if __name__ == "__main__":
    from app.agent.search_agent_V2 import run_agent

    query = (
        "Detail the performance benchmarks of Retrieval-Aware Fine-Tuning (RAFT) "
        "techniques compared to standard RAG pipelines in recent domain-specific evaluations."
    )
    result, raw = run_agent(query, "963okmjnbvcx")

    print("raw content length:", len(raw))
    raw_clean_dict = clean(raw)
    print("cleaned length:", len(raw_clean_dict))
