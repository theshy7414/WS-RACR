"""
pytest 單元測試：InteractionLog 與 InteractionRecord
涵蓋 record / to_context_text / maxlen 截斷三個核心行為。
"""
import sys
import os
import time

from ws_racr.memory import InteractionRecord, InteractionLog


def _make_record(utterance: str, response_type: str = "execute", step: str = "idle") -> InteractionRecord:
    """建立固定 timestamp 的測試用 InteractionRecord。"""
    return InteractionRecord(
        timestamp=1000.0,  # 固定值，deterministic
        user_utterance=utterance,
        response_type=response_type,
        robot_response="ok",
        step_context=step,
    )


def test_interaction_log_record_and_len():
    """測試 record() 後 __len__ 正確增加。"""
    log = InteractionLog(maxlen=20)
    assert len(log) == 0
    log.record(_make_record("幫我拿扳手", "execute"))
    assert len(log) == 1
    log.record(_make_record("第三步需要什麼？", "answer"))
    assert len(log) == 2


def test_interaction_log_to_context_text_returns_empty_when_no_records():
    """空日誌 to_context_text() 應回傳空字串，不應注入任何 section。"""
    log = InteractionLog(maxlen=20)
    result = log.to_context_text(n=5)
    assert result == ""


def test_interaction_log_to_context_text_contains_utterance():
    """to_context_text() 輸出應包含 user_utterance 內容。"""
    log = InteractionLog(maxlen=20)
    log.record(_make_record("把螺絲給我", "execute", step="tighten_small_screw_under_platform"))
    text = log.to_context_text(n=5)
    assert "把螺絲給我" in text


def test_interaction_log_to_context_text_respects_n():
    """to_context_text(n=2) 只應包含最近 2 筆，不包含較舊的記錄。"""
    log = InteractionLog(maxlen=20)
    log.record(_make_record("舊的指令A"))
    log.record(_make_record("中間指令B"))
    log.record(_make_record("最新指令C"))
    text = log.to_context_text(n=2)
    assert "最新指令C" in text
    assert "中間指令B" in text
    assert "舊的指令A" not in text


def test_interaction_log_maxlen_truncation():
    """超過 maxlen 時最舊的記錄應被丟棄（FIFO）。"""
    log = InteractionLog(maxlen=3)
    for i in range(4):
        log.record(_make_record(f"指令{i}"))
    assert len(log) == 3
    # 最舊的「指令0」應已被擠出
    text = log.to_context_text(n=10)
    assert "指令0" not in text
    assert "指令3" in text


def test_interaction_record_to_text_format():
    """InteractionRecord.to_text() 應包含 response_type 大寫標籤與 utterance。"""
    rec = _make_record("請問現在第幾步？", "answer", step="place_platform")
    text = rec.to_text()
    assert "[ANSWER]" in text
    assert "請問現在第幾步？" in text
    assert "place_platform" in text
