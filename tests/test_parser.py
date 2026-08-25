# tests/test_parser.py
"""
Unit tests for app/agent/parser.py

Pure-function tests: no network, no DB, no mocking needed.
Run with: pytest tests/test_parser.py -v
"""

from app.agent.parser import parse_agent_output


class TestEmptyInput:
    def test_no_search_results_key(self):
        sources, raw_docs = parse_agent_output({})
        assert sources == []
        assert raw_docs == []

    def test_empty_search_results_list(self):
        sources, raw_docs = parse_agent_output({"search_results": []})
        assert sources == []
        assert raw_docs == []

    def test_query_block_with_no_results_key(self):
        agent_output = {"search_results": [{"query": "q1", "error": "timeout"}]}
        sources, raw_docs = parse_agent_output(agent_output)
        assert sources == []
        assert raw_docs == []


class TestReturnShape:
    def test_returns_tuple_of_two_lists(self):
        agent_output = {
            "search_results": [
                {
                    "query": "q1",
                    "results": [
                        {"url": "https://a.com", "title": "A", "content": "text a"}
                    ],
                }
            ]
        }
        result = parse_agent_output(agent_output)
        assert isinstance(result, tuple)
        assert len(result) == 2
        sources, raw_docs = result
        assert isinstance(sources, list)
        assert isinstance(raw_docs, list)

    def test_raw_docs_have_expected_keys(self):
        agent_output = {
            "search_results": [
                {
                    "query": "q1",
                    "results": [
                        {"url": "https://a.com", "title": "A", "content": "text a"}
                    ],
                }
            ]
        }
        _, raw_docs = parse_agent_output(agent_output)
        assert set(raw_docs[0].keys()) == {"url", "title", "content"}


class TestDeduplication:
    def test_duplicate_url_across_queries_kept_once(self):
        agent_output = {
            "search_results": [
                {
                    "query": "q1",
                    "results": [
                        {"url": "https://a.com", "title": "A", "content": "text a"}
                    ],
                },
                {
                    "query": "q2",
                    "results": [
                        {
                            "url": "https://a.com",
                            "title": "A dup",
                            "content": "text a dup",
                        }
                    ],
                },
            ]
        }
        sources, raw_docs = parse_agent_output(agent_output)
        assert sources == ["https://a.com"]
        assert len(raw_docs) == 1

    def test_distinct_urls_all_kept(self):
        agent_output = {
            "search_results": [
                {
                    "query": "q1",
                    "results": [
                        {"url": "https://a.com", "title": "A", "content": "text a"},
                        {"url": "https://b.com", "title": "B", "content": "text b"},
                    ],
                }
            ]
        }
        sources, raw_docs = parse_agent_output(agent_output)
        assert sources == ["https://a.com", "https://b.com"]
        assert len(raw_docs) == 2

    def test_empty_url_skipped(self):
        agent_output = {
            "search_results": [
                {
                    "query": "q1",
                    "results": [
                        {"url": "", "title": "No URL", "content": "text"},
                        {"url": "https://a.com", "title": "A", "content": "text a"},
                    ],
                }
            ]
        }
        sources, raw_docs = parse_agent_output(agent_output)
        assert sources == ["https://a.com"]
        assert len(raw_docs) == 1


class TestMissingFields:
    def test_missing_title_defaults_empty_string(self):
        agent_output = {
            "search_results": [
                {"query": "q1", "results": [{"url": "https://a.com", "content": "x"}]}
            ]
        }
        _, raw_docs = parse_agent_output(agent_output)
        assert raw_docs[0]["title"] == ""

    def test_missing_content_defaults_empty_string(self):
        agent_output = {
            "search_results": [
                {"query": "q1", "results": [{"url": "https://a.com", "title": "A"}]}
            ]
        }
        _, raw_docs = parse_agent_output(agent_output)
        assert raw_docs[0]["content"] == ""

    def test_non_dict_query_block_ignored(self):
        agent_output = {"search_results": ["not a dict"]}
        sources, raw_docs = parse_agent_output(agent_output)
        assert sources == []
        assert raw_docs == []


class TestOrderPreserved:
    def test_source_order_matches_first_seen(self):
        agent_output = {
            "search_results": [
                {
                    "query": "q1",
                    "results": [
                        {"url": "https://b.com", "title": "B", "content": "text b"},
                        {"url": "https://a.com", "title": "A", "content": "text a"},
                    ],
                }
            ]
        }
        sources, _ = parse_agent_output(agent_output)
        assert sources == ["https://b.com", "https://a.com"]
