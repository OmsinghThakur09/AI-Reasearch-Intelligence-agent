# tests/test_pipeline.py
"""
Integration test for app/agent/pipeline.py

Mocks out every external dependency (Tavily/agent, DB, vectorstore, LLM)
so this test runs fast and offline, but exercises the full
stream_research_pipeline() wiring end-to-end — the same thing the build
plan calls "the integration test" for Week 3.

Everything below is patched at the app.agent.pipeline module namespace,
since that's where these names were imported into (not their original
defining module) — patching the origin module would not affect calls
made from inside pipeline.py.

Run with: pytest tests/test_pipeline.py -v
"""

from unittest.mock import patch, MagicMock
import uuid

import app.agent.pipeline as pipeline

FAKE_QUERY_ID = uuid.uuid4()


def _fake_agent_output():
    "minimal agent_output shaped like what search_agent.run_agent() returns"
    agent_output = {
        "search_results": [
            {
                "query": "test query",
                "results": [
                    {
                        "url": "https://example.com/article",
                        "title": "Example Article",
                        "content": "Some example content about the topic being researched.",
                    }
                ],
            }
        ]
    }
    raw_content = []  # no escalated full-page fetches in this run
    sub_queries = ["test query"]
    return agent_output, raw_content, sub_queries


def _fake_clean(texts):
    "stand-in for app.utils.cleaner.clean(): keep every doc, tag with og_idx"
    return [{"og_idx": i, "raw": t, "clean": t} for i, t in enumerate(texts) if t]


def _fake_stream_chunks():
    "stand-in for rag_chain.stream(): yields plain string tokens"
    yield "The answer "
    yield "mentions the example topic."


@patch("app.agent.pipeline.save_user_query")
@patch("app.agent.pipeline.save_documents")
@patch("app.agent.pipeline.save_sources")
@patch("app.agent.pipeline.update_query_status")
@patch("app.agent.pipeline.save_agent_actions")
@patch("app.agent.pipeline.update_error_message")
@patch("app.agent.pipeline.run_agent")
@patch("app.agent.pipeline.clean")
@patch("app.agent.pipeline.ingest_clean_text")
@patch("app.agent.pipeline.retrieve_by_subqueries")
@patch("app.agent.pipeline.build_llm_call")
def test_pipeline_completes_and_saves_expected_rows(
    mock_build_llm_call,
    mock_retrieve_by_subqueries,
    mock_ingest_clean_text,
    mock_clean,
    mock_run_agent,
    mock_update_error_message,
    mock_save_agent_actions,
    mock_update_query_status,
    mock_save_sources,
    mock_save_documents,
    mock_save_user_query,
):
    # --- arrange ---
    mock_save_user_query.return_value = FAKE_QUERY_ID
    mock_run_agent.return_value = _fake_agent_output()
    mock_clean.side_effect = _fake_clean
    mock_retrieve_by_subqueries.return_value = "fake retrieved context"

    mock_chain = MagicMock()
    mock_chain.stream.return_value = _fake_stream_chunks()
    mock_build_llm_call.return_value = mock_chain

    # --- act ---
    result = pipeline.run_research_pipeline("test query", session_id=None)

    # --- assert: the three checks the build plan's integration test calls for ---

    # 1. a "queries" row hits status='completed'
    mock_update_query_status.assert_called_once_with(FAKE_QUERY_ID, "completed")

    # 2. at least one "documents" row has non-null clean_text
    assert mock_save_documents.call_count == 1
    _, docs_arg, clean_dict_arg = mock_save_documents.call_args[0]
    assert len(clean_dict_arg) >= 1
    assert all(row["clean"] for row in clean_dict_arg)

    # 3. at least one "sources" row exists
    assert mock_save_sources.call_count == 1
    _, sources_arg = mock_save_sources.call_args[0]
    assert len(sources_arg) >= 1

    # sanity on the final streamed answer / return shape
    assert result is not None
    assert result["answer"] == "The answer mentions the example topic."
    assert result["sources"] == ["https://example.com/article"]
    assert result["query_id"] == str(FAKE_QUERY_ID)

    # error path should never fire on the happy path
    mock_update_error_message.assert_not_called()


@patch("app.agent.pipeline.save_user_query")
@patch("app.agent.pipeline.update_query_status")
@patch("app.agent.pipeline.update_error_message")
@patch("app.agent.pipeline.run_agent")
def test_pipeline_marks_failed_on_exception(
    mock_run_agent,
    mock_update_error_message,
    mock_update_query_status,
    mock_save_user_query,
):
    # --- arrange: force the agent step itself to blow up ---
    mock_save_user_query.return_value = FAKE_QUERY_ID
    mock_run_agent.side_effect = RuntimeError("search backend unreachable")

    # --- act / assert ---
    events = []
    try:
        for event in pipeline.stream_research_pipeline("test query", session_id=None):
            events.append(event)
    except RuntimeError:
        pass  # stream_research_pipeline re-raises after yielding the error event

    # a "queries" row should be marked failed with the error message recorded
    mock_update_query_status.assert_called_once_with(FAKE_QUERY_ID, "failed")
    mock_update_error_message.assert_called_once_with(
        FAKE_QUERY_ID, "search backend unreachable"
    )

    error_events = [e for e in events if e["event"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["data"] == "search backend unreachable"
