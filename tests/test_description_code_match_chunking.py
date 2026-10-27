from types import SimpleNamespace
from unittest.mock import Mock

import ai.description_code_match_validator as validator


def _completion(text, usage=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )


def _install_mock_client(monkeypatch, side_effect=None, response=None):
    client = Mock()
    if side_effect is not None:
        client.chat.completions.create.side_effect = side_effect
    else:
        client.chat.completions.create.return_value = response or _completion(
            "Verdict: PASS\nReason: The described changes are supported."
        )
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(validator, "OpenAI", Mock(return_value=client))
    # A deterministic test estimator; production uses the shared project helper.
    monkeypatch.setattr(validator, "count_tokens", lambda text: (len(text) + 7) // 8)
    return client


def _file_diff(path, marker="value = 1", lines=1):
    changed_lines = "".join(f"+{marker}_{index} = {index}\n" for index in range(lines))
    return (
        f"diff --git a/{path} b/{path}\n"
        f"index 1111111..2222222 100644\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"@@ -1,1 +1,{lines} @@\n"
        f"{changed_lines}"
    )


def test_small_single_file_diff_validates_directly(monkeypatch):
    client = _install_mock_client(monkeypatch)
    passed, reason = validator.validate_description_matches_code(
        "Add validation", "Add input validation.", _file_diff("validation.py")
    )

    assert passed is True
    assert reason.startswith("PASS:")
    assert client.chat.completions.create.call_count == 1


def test_multiple_small_files_are_sent_together_once(monkeypatch):
    client = _install_mock_client(monkeypatch)
    combined_diff = _file_diff("auth.py") + _file_diff("analytics.py")

    passed, reason = validator.validate_description_matches_code(
        "Add auth and analytics",
        "Adds authentication and analytics tracking.",
        combined_diff,
    )

    assert passed is True
    assert reason.startswith("PASS:")
    assert client.chat.completions.create.call_count == 1
    sent_prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "auth.py" in sent_prompt
    assert "analytics.py" in sent_prompt


def test_large_diff_is_split_and_all_passing_chunks_aggregate_to_pass(monkeypatch):
    client = _install_mock_client(monkeypatch)
    monkeypatch.setattr(validator, "SAFE_INPUT_TOKEN_LIMIT", 900)
    large_diff = "".join(
        _file_diff(f"module_{file_index}.py", marker="business_logic", lines=180)
        for file_index in range(4)
    )

    passed, reason = validator.validate_description_matches_code(
        "Update modules", "Updates the module implementations.", large_diff
    )

    assert passed is True
    assert reason.startswith("PASS:")
    assert client.chat.completions.create.call_count > 1
    for call in client.chat.completions.create.call_args_list:
        prompt = call.kwargs["messages"][0]["content"]
        assert validator.count_tokens(prompt) <= validator.SAFE_INPUT_TOKEN_LIMIT


def test_mismatch_in_later_chunk_makes_overall_result_fail(monkeypatch):
    def response_for_prompt(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "module_3.py" in prompt:
            return _completion(
                "Verdict: FAIL\nReason: The documented behavior is contradicted in this module."
            )
        return _completion("Verdict: PASS\nReason: No contradiction in this section.")

    client = _install_mock_client(monkeypatch, side_effect=response_for_prompt)
    monkeypatch.setattr(validator, "SAFE_INPUT_TOKEN_LIMIT", 900)
    large_diff = "".join(
        _file_diff(f"module_{file_index}.py", marker="business_logic", lines=180)
        for file_index in range(4)
    )

    passed, reason = validator.validate_description_matches_code(
        "Update modules", "Updates all module implementations.", large_diff
    )

    assert passed is False
    assert reason.startswith("FAIL:")
    assert "contradicted" in reason
    assert client.chat.completions.create.call_count > 1


def test_api_failure_in_a_chunk_is_error_and_does_not_short_circuit(monkeypatch):
    counter = {"calls": 0}

    def response_with_one_failure(**_kwargs):
        counter["calls"] += 1
        if counter["calls"] == 2:
            raise RuntimeError("temporary API error")
        return _completion("Verdict: PASS\nReason: No mismatch found.")

    client = _install_mock_client(monkeypatch, side_effect=response_with_one_failure)
    monkeypatch.setattr(validator, "SAFE_INPUT_TOKEN_LIMIT", 900)
    large_diff = "".join(
        _file_diff(f"module_{file_index}.py", marker="business_logic", lines=180)
        for file_index in range(4)
    )

    passed, reason = validator.validate_description_matches_code(
        "Update modules", "Updates the module implementations.", large_diff
    )

    assert passed is False
    assert reason.startswith("ERROR:")
    assert "Chunk 2" in reason
    assert counter["calls"] > 2
    assert client.chat.completions.create.call_count == counter["calls"]


def test_malformed_response_is_error_not_pass(monkeypatch):
    client = _install_mock_client(
        monkeypatch,
        response=_completion("The changes seem okay, but no verdict was provided."),
    )

    passed, reason = validator.validate_description_matches_code(
        "Add validation", "Adds input validation.", _file_diff("validation.py")
    )

    assert passed is False
    assert reason.startswith("ERROR:")
    assert "unrecognized verdict" in reason
    assert client.chat.completions.create.call_count == 1


def test_description_covering_only_one_unrelated_change_can_fail(monkeypatch):
    def response_for_prompt(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "analytics.py" in prompt:
            return _completion(
                "Verdict: FAIL\nReason: The analytics change is unrelated to the described auth change."
            )
        return _completion("Verdict: PASS\nReason: The auth change matches the description.")

    client = _install_mock_client(monkeypatch, side_effect=response_for_prompt)
    combined_diff = _file_diff("auth.py") + _file_diff("analytics.py")

    passed, reason = validator.validate_description_matches_code(
        "Add authentication", "Adds authentication checks.", combined_diff
    )

    assert passed is False
    assert reason.startswith("FAIL:")
    assert "analytics change is unrelated" in reason
    assert client.chat.completions.create.call_count == 1


def test_empty_description_is_skipped_not_pass(monkeypatch):
    passed, reason = validator.validate_description_matches_code(
        "Title", "", _file_diff("a.py")
    )
    assert passed is False
    assert reason.startswith("SKIPPED:")
