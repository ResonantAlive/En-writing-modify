import pytest

from app.grading.classify import api_outcome_for_breaker, classify_call
from app.grading.parse import parse_response
from app.providers.client import RawCall

KW = dict(score_key="判分", score_aliases=["分数", "score"], criteria_key="判据",
          required_keys=["判分", "判据"], full_score=10)


def P(text, **over):
    return parse_response(text, **{**KW, **over})


GOOD = '{"判分": 7, "判据": [{"name": "内容", "body": "好"}], "评语": "不错"}'


def test_direct():
    r = P(GOOD)
    assert (r.level, r.error_code, r.score) == ("direct", None, 7.0)
    assert r.obj["评语"] == "不错"


def test_fenced_json_and_plain_fence():
    r = P("```json\n" + GOOD + "\n```")
    assert (r.level, r.error_code) == ("fenced", None)
    r = P("```\n" + GOOD + "\n```")
    assert (r.level, r.error_code) == ("fenced", None)


def test_prefix_suffix_text():
    r = P("结果如下：\n" + GOOD + "\n谢谢")
    assert (r.level, r.error_code, r.score) == ("balanced", None, 7.0)


def test_nested_braces_inside_strings():
    t = '前言 {"判分": 8, "判据": [{"name": "a}b{", "body": "含有 { 大括号 } 和 \\"引号\\""}], "x": {"y": 1}} 后记 {'
    r = P(t)
    assert r.error_code is None and r.level == "balanced"
    assert r.obj["判据"][0]["name"] == "a}b{" and r.obj["x"] == {"y": 1}


def test_trailing_comma_keeps_all_fields():
    t = '{"判分": 7, "判据": [{"name": "内容", "body": "切题"},], "评语": "好", "extra": {"a": [1,2,],},}'
    r = P(t)
    assert r.level == "repair" and r.error_code is None and r.score == 7
    assert r.obj["评语"] == "好" and r.obj["extra"] == {"a": [1, 2]} and r.obj["判据"][0]["body"] == "切题"


def test_single_quotes():
    r = P("{'判分': 6, '判据': [{'name': 'x', 'body': 'y'}]}")
    assert r.level == "repair" and r.error_code is None and r.score == 6
    assert r.obj["判据"] == [{"name": "x", "body": "y"}]


def test_comments():
    r = P('{"判分": 5, // the score\n "判据": [] /* none */, "n": "k"}')
    assert r.error_code is None and r.score == 5 and r.obj["n"] == "k"


def test_raw_newline_in_string():
    r = P('{"判分": 5, "判据": [{"name": "a", "body": "line1\nline2"}]}')
    assert r.error_code is None and "line2" in r.obj["判据"][0]["body"]


def test_python_literals():
    r = P('{"判分": 5, "判据": [], "ok": True, "none": None, "f": False}')
    assert r.error_code is None
    assert r.obj["ok"] is True and r.obj["none"] is None and r.obj["f"] is False


def test_truncated_json_is_repaired_but_validated():
    r = P('{"判分": 7, "判据": [{"name": "内容", "body": "切')
    assert r.level == "repair"
    assert r.error_code is None and r.score == 7
    r = P('{"评语": "好", "判分": 7')            # truncated before 判据
    assert r.level == "repair" and r.error_code == "missing_required_field"
    assert r.obj["评语"] == "好"


def test_missing_field():
    r = P('{"判分": 7}')
    assert r.error_code == "missing_required_field" and r.obj == {"判分": 7} and r.level == "direct"
    r = P('{"判据": []}')
    assert r.error_code == "missing_required_field"


def test_score_aliases_and_numeric_strings():
    r = P('{"分数": "8", "判据": []}')
    assert r.error_code is None and r.score == 8 and r.obj["判分"] == "8" and r.obj["分数"] == "8"
    r = P('{"判分": "8/10", "判据": []}')
    assert r.error_code == "missing_required_field"      # not numeric
    r = P('{"判分": true, "判据": []}')
    assert r.error_code == "missing_required_field"


def test_score_out_of_range():
    assert P('{"判分": 11, "判据": []}').error_code == "score_out_of_range"
    assert P('{"判分": -1, "判据": []}').error_code == "score_out_of_range"
    assert P('{"判分": 10, "判据": []}').error_code is None
    assert P('{"判分": 0, "判据": []}').error_code is None
    assert P('{"判分": 7.5, "判据": []}').score == 7.5


def test_non_json_and_empty_and_refusal():
    assert P("这篇作文不错").error_code == "json_parse_failed"
    assert P("").error_code == "empty_response"
    assert P("   \n ").error_code == "empty_response"
    assert P(None).error_code == "empty_response"
    r = P("I'm sorry, but I can't help with that request.")
    assert r.error_code == "refusal_or_safety"
    assert P("抱歉，我无法协助完成该请求。").error_code == "refusal_or_safety"


def test_regex_only_is_missing_field_with_score():
    r = P("判分: 7")
    assert (r.level, r.error_code, r.score) == ("regex", "missing_required_field", 7.0)
    r = P("最终得分，分数：8.5 分")
    assert r.level == "regex" and r.score == 8.5
    r = P("判分: 77")
    assert r.level == "regex" and r.score is None and r.error_code == "score_out_of_range"


def test_regex_ok_when_only_score_required():
    r = P("判分: 7", required_keys=["判分"])
    assert (r.level, r.error_code, r.score) == ("regex", None, 7.0)


def test_list_top_level_is_not_an_object():
    r = P('[1, 2, 3]')
    assert r.error_code == "json_parse_failed"


def test_largest_object_wins():
    t = '{"a": 1} then ' + GOOD
    r = P(t)
    assert r.error_code is None and r.score == 7


# ---------------------------------------------------------------- classification table
def raw(status=None, kind=None):
    return RawCall(http_status=status, body=None, text="", latency_ms=1, error_kind=kind)


@pytest.mark.parametrize("st", [400, 405, 409, 413, 415, 418, 422, 451, 499])
def test_class_c(st):
    assert classify_call(raw(st)) == "C"
    assert api_outcome_for_breaker("C") == "other"


@pytest.mark.parametrize("st", [401, 402, 403, 404, 408, 429, 500, 502, 503, 504, 599, 301, 302, 307, 100, 199, 600])
def test_class_b_status(st):
    assert classify_call(raw(st)) == "B"


@pytest.mark.parametrize("kind", ["timeout", "connect", "network"])
def test_class_b_transport(kind):
    assert classify_call(raw(None, kind)) == "B"
    assert api_outcome_for_breaker("B") == "api_error"


@pytest.mark.parametrize("st", [200, 201, 204, 299])
def test_ok(st):
    assert classify_call(raw(st)) == "ok"


def test_breaker_outcome_2xx_always_ok():
    assert api_outcome_for_breaker("ok") == "ok"
    assert api_outcome_for_breaker("A") == "ok"      # 2xx with unusable content still resets the streak
