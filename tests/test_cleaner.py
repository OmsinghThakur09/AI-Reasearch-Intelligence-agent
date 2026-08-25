# tests/test_cleaner.py
"""
Unit tests for app/utils/cleaner.py

These are pure-function tests: no network, no DB, no mocking needed.
Run with: pytest tests/test_cleaner.py -v
"""

from app.utils.cleaner import clean, MIN_WORD_COUNT

# NOTE: these tests assume clean([]) returns [] rather than raising —
# requires the empty-input guard added to cleaner.py (see clean() docstring
# / top of function body). If that guard isn't in place yet, the very first
# test below will fail with an AttributeError from pandas.


def _long_enough(text: str) -> str:
    """Pad a short string with generic filler words so it clears
    MIN_WORD_COUNT with comfortable margin.

    IMPORTANT: the filler must NOT be a repeated phrase. cleaner.py's own
    duplicate-consecutive-phrase regex (`re.sub(r"\\b(.{5,80}?)\\b\\s+\\1\\b", ...)`)
    collapses repeated substrings, so e.g. "lorem ipsum " * 5 gets silently
    shrunk back down below the word-count threshold — that was the actual
    cause of the earlier IndexError/assert-0 failures, not fixture length.
    Using a single non-repeating sentence sidesteps that entirely.
    """
    filler = (
        " the quick brown fox jumps over a lazy dog near the river bank"
        " while clouds drift slowly across an open afternoon sky above"
        " distant hills covered in green pine trees and scattered rocks"
        " as a gentle breeze moves through tall grass fields nearby town"
    )
    return f"{text} {filler}"


class TestBasicCleaning:
    def test_empty_list_returns_empty(self):
        # clean([]) must return [] cleanly rather than raising — an empty
        # batch (e.g. zero search results) is a normal, expected input,
        # not a caller error.
        assert clean([]) == []

    def test_none_values_handled(self):
        result = clean([None, _long_enough("Real content about a topic.")])
        # None should not crash and should not produce a row (too short / empty)
        assert all(row["clean"] != "" for row in result)

    def test_whitespace_normalized(self):
        raw = _long_enough("This   has\n\nweird   whitespace.")
        result = clean([raw])
        assert len(result) == 1
        assert "   " not in result[0]["clean"]

    def test_og_idx_preserved(self):
        docs = [
            _long_enough("First document about renewable energy."),
            _long_enough("Second document about quantum computing."),
        ]
        result = clean(docs)
        idxs = {row["og_idx"] for row in result}
        assert idxs == {0, 1}


class TestBoilerplateRemoval:
    def test_subscribe_line_removed(self):
        raw = _long_enough("Real article content here.") + "\nSubscribe\n"
        result = clean([raw])
        assert "Subscribe" not in result[0]["clean"]

    def test_citation_lines_removed(self):
        raw = (
            _long_enough("Findings about the study are described in detail below.")
            + "\nSmith J. doi: 10.1234/abcd.5678\n[PMC free article]\n[PubMed]\n"
        )
        result = clean([raw])
        assert "doi:" not in result[0]["clean"]
        assert "PMC free article" not in result[0]["clean"]

    def test_boilerplate_heading_caught(self):
        raw = _long_enough("Body text about a topic that matters.") + "\n### Search\n"
        result = clean([raw])
        assert "### Search" not in result[0]["clean"]

    def test_real_content_not_removed(self):
        raw = _long_enough(
            "Resourceful engineers often explore new solutions to hard problems."
        )
        result = clean([raw])
        # "Resourceful" and "explore new solutions" should survive even though
        # "Resources" and "Explore .*" are boilerplate patterns — those patterns
        # only match whole lines, not substrings within real sentences.
        assert "Resourceful" in result[0]["clean"]


class TestRepeatedShortLines:
    def test_repeated_short_line_stripped(self):
        raw = _long_enough("Main body content discussing the topic in depth.")
        raw += "\nJohnDoe123\nJohnDoe123\nJohnDoe123\n"
        result = clean([raw])
        # only first occurrence should remain
        assert result[0]["clean"].count("JohnDoe123") <= 1

    def test_short_line_below_threshold_kept(self):
        raw = _long_enough("Main body content discussing the topic in depth.")
        raw += "\nUniqueLineA\nUniqueLineB\n"
        result = clean([raw])
        assert "UniqueLineA" in result[0]["clean"]


class TestArtifactStripping:
    def test_truncation_marker_removed(self):
        raw = _long_enough("Some content before the marker [...] and after it.")
        result = clean([raw])
        assert "[...]" not in result[0]["clean"]


class TestWordCountFiltering:
    def test_short_document_dropped(self):
        result = clean(["Too short."])
        assert result == []

    def test_document_over_threshold_kept(self):
        raw = _long_enough("This document has plenty of real content to keep.")
        assert len(raw.split()) > MIN_WORD_COUNT
        result = clean([raw])
        assert len(result) == 1

    def test_mixed_short_and_long_docs(self):
        docs = [
            "Way too short.",
            _long_enough("This one is definitely long enough to survive filtering."),
        ]
        result = clean(docs)
        assert len(result) == 1
        assert result[0]["og_idx"] == 1


class TestDeduplication:
    def test_exact_duplicates_collapsed(self):
        doc = _long_enough("Duplicate content appearing twice in the batch.")
        result = clean([doc, doc])
        assert len(result) == 1

    def test_near_duplicates_collapsed(self):
        base = _long_enough(
            "Renewable energy adoption has accelerated across many countries this year"
        )
        # same opening, different tail -> should be treated as near-duplicate
        variant = base + " with some extra unique trailing sentence appended."
        result = clean([base, variant])
        assert len(result) == 1

    def test_distinct_documents_both_kept(self):
        docs = [
            _long_enough("Article about solar panel efficiency improvements."),
            _long_enough("Article about a completely unrelated topic like jazz."),
        ]
        result = clean(docs)
        assert len(result) == 2


class TestOutputShape:
    def test_output_has_expected_keys(self):
        raw = _long_enough("Content with expected output shape for this test.")
        result = clean([raw])
        assert set(result[0].keys()) == {"raw", "og_idx", "clean"}
