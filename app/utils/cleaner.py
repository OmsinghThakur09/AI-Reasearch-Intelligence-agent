# app/utils/cleaner.py
"""
function to clean raw web results from tavily search:
    Handle None values
    Strip whitespace
    Normalize whitespace/newlines
    Remove common website boilerplate (line-anchored, not open-ended)
    Remove repeated headings
    Remove short lines that repeat many times in one doc (usernames, nav spam)
    Strip mid-text artifacts (Tavily's own "[...]" truncation marker)
    Drop very short documents (by real word count, measured AFTER cleaning)
    Drop duplicate / near-duplicate documents
    Return cleaned documents

Patterns below were built against real Tavily search_node output spanning
several very different query types (consumer trend lists, forum threads,
academic papers, biomedical literature with citation lists, tech blogs).
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
    # --- added from real-world Tavily samples ---
    r"[\d,]+ character limit reached\.?",
    r"Refer to caption",
    r"Press enter or click to view image in full size",
    r"Image \d+.*",
    r"Join Medium for free to get updates from this writer\.?",
    r"Subscribe",
    r"Remember me for faster sign in",
    r"-\s*\[x\]\s*",
    r"(All )?Topics",
    r"Featured Case Studies",
    r"Site Links",
    r"Advertise with Us.*",
    r"Community Rules.*",
    r"Manage Email Subscriptions.*",
    r"Premium Pilot Upgrades",
    r"Similar threads",
    r"Members online",
    r"Forum statistics",
    r"About us",
    r"New Threads",
    r"Search",
    r"LoginPricing",
    r"We value your privacy.*",
    r"Advertisement\b.*",
    r"Previous Close.*",
    r"Volume\s+[\d,]+.*",
    r".*\bdoi:\s*10\.\d{4,9}/\S*.*",
    r".*\[PMC free article\].*",
    r".*\[PubMed\].*",
    r".*\[Google Scholar\].*",
]

_BOILERPLATE_RE = [
    re.compile(p, flags=re.IGNORECASE) for p in BOILERPLATE_LINE_PATTERNS
]

# Mid-text artifacts that show up INSIDE a paragraph rather than as their
# own line, so the line-based boilerplate filter above can't catch them.
# Tavily's own "[...]" truncation marker showed up in every sample
# reviewed, regardless of query topic or source site.
ARTIFACT_PATTERNS = [
    re.compile(r"\[\.\.\.\]"),
]

MIN_WORD_COUNT = 40  # real word count, measured AFTER boilerplate/artifact stripping
NEAR_DUP_PREFIX_WORDS = 40  # words compared when checking near-duplicates

# a short line that repeats this often within ONE document is almost never
# real content — it's forum usernames, nav items, or repeated UI text
# (e.g. "Subscribe" appearing twice, a username appearing ten times) that
# slipped past the fixed boilerplate pattern list above.
REPEATED_LINE_MAX_WORDS = 4
REPEATED_LINE_MIN_COUNT = 3


def _is_boilerplate_line(line: str) -> bool:
    "return True if a single line is entirely a boilerplate/nav/footer line"
    stripped = line.strip()
    if not stripped:
        return True

    # Check for markdown table rows (starts with or contains multiple pipes)
    if "|" in stripped and stripped.count("|") > 2:
        return True
    if stripped.startswith("---") or stripped.startswith("|"):
        return True

    # strip markdown heading markers ("#", "##", "###" ...) before matching,
    # so a boilerplate phrase used as a heading (e.g. "### Search") is still
    # caught, and a heading that's just bare hashes counts as blank too.
    check_line = re.sub(r"^#{1,6}\s*", "", stripped)
    if not check_line:
        return True

    for pattern in _BOILERPLATE_RE:
        if pattern.fullmatch(check_line):
            return True
    return False


def _strip_boilerplate(text: str) -> str:
    "drop boilerplate lines individually, keep every real content line intact"
    lines = text.split("\n")
    kept = [ln for ln in lines if not _is_boilerplate_line(ln)]
    return "\n".join(kept)


def _strip_repeated_short_lines(text: str) -> str:
    """
    Drop short lines (<= REPEATED_LINE_MAX_WORDS words) that repeat
    REPEATED_LINE_MIN_COUNT+ times in the same document. Keeps the first
    occurrence (in case it's meaningful) and removes the rest, catching
    spam patterns (usernames, repeated nav items) too varied for a fixed
    pattern list to anticipate.
    """
    lines = text.split("\n")
    counts: dict[str, int] = {}
    for ln in lines:
        stripped = ln.strip()
        if stripped and len(stripped.split()) <= REPEATED_LINE_MAX_WORDS:
            counts[stripped] = counts.get(stripped, 0) + 1

    spam_lines = {k for k, v in counts.items() if v >= REPEATED_LINE_MIN_COUNT}

    kept = []
    seen_spam = set()
    for ln in lines:
        stripped = ln.strip()
        if stripped in spam_lines:
            if stripped in seen_spam:
                continue
            seen_spam.add(stripped)
        kept.append(ln)
    return "\n".join(kept)


def _strip_artifacts(text: str) -> str:
    "remove mid-text artifacts (e.g. Tavily's '[...]' truncation marker)"
    for pattern in ARTIFACT_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def _word_count(text: str) -> int:
    return len(text.split())


def _near_dup_key(text: str) -> str:
    "cheap near-duplicate signature: normalized first N words"
    words = re.sub(r"\s+", " ", text).strip().lower().split()
    return " ".join(words[:NEAR_DUP_PREFIX_WORDS])


def clean(raw_texts: list[str]) -> list[dict]:
    if not raw_texts:
        return []

    df = pd.DataFrame({"raw": raw_texts})

    # replace none with empty string
    df["raw"] = df["raw"].fillna("")

    # strip leading/trailing whitespace on the raw blob
    df["clean"] = df["raw"].str.strip()

    # line-based passes happen BEFORE whitespace is collapsed, so line
    # boundaries are still intact to check each line individually
    df["clean"] = df["clean"].apply(_strip_boilerplate)
    df["clean"] = df["clean"].apply(_strip_repeated_short_lines)

    # remove mid-paragraph artifacts (Tavily's own truncation marker, etc.)
    df["clean"] = df["clean"].apply(_strip_artifacts)

    # Ensure broken sentences ending in a newline get a period before joining,
    # and replace newlines with a space to prevent word-gluing.
    df["clean"] = df["clean"].apply(lambda x: re.sub(r"([a-zA-Z0-9])\n", r"\1. ", x))

    # normalize newlines and whitespace (safe to collapse now)
    df["clean"] = df["clean"].apply(lambda x: re.sub(r"\s+", " ", x))

    # remove duplicate consecutive headings/phrases
    df["clean"] = df["clean"].apply(
        lambda x: re.sub(r"\b(.{5,80}?)\b\s+\1\b", r"\1", x, flags=re.IGNORECASE)
    )

    df["clean"] = df["clean"].str.strip()

    # drop very short entries using real word count, not character count —
    # measured AFTER cleaning, so this now catches docs that are junk-free
    # but too thin to be useful (e.g. a reference list that stripped down
    # to nothing), not just short snippets in general
    df["word_count"] = df["clean"].apply(_word_count)

    df = df[df["word_count"] > MIN_WORD_COUNT]

    # drop exact duplicates
    df = df.drop_duplicates(subset="clean")

    # drop near-duplicates (e.g. same article fetched twice via two
    # differently-worded search queries, so text isn't byte-identical
    # but opens with the same content)
    df["near_dup_key"] = df["clean"].apply(_near_dup_key)
    df = df.drop_duplicates(subset="near_dup_key")

    df = df.drop(columns=["word_count", "near_dup_key"])

    # return list of {'og_idx':..., 'raw':..., 'clean':...}
    return df.to_dict("records")


# if __name__ == "__main__":
#     from app.agent.search_agent import run_agent
#     from app.agent.parser import parse_agent_output

#     query = (
#         r"Inform me about the latest high-temperature superconductor breakthroughs and the specific transition temperatures ((T_{c})) achieved in recent 2025 or 2026 preprints"
#     )
#     result, raw, _ = run_agent(query, "9865oikl7834dyho")

#     _, metadata = parse_agent_output(result)
#     raw_clean_dict = clean([row["content"] for row in metadata])
#     print(len(raw_clean_dict))
#     print(len(metadata))
