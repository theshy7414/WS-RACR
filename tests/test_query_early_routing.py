"""
pytest 單元測試：execute_task() 早期查詢路由（Step 2.8 前）

測試 _is_query_intent() 新增模式、action-based 備援、
以及 _handle_query_via_voice_query() 的 InteractionLog 寫入。
"""
import sys
import os
from unittest.mock import MagicMock, patch

# 最小化 import—只測試方法邏輯，不需要真實 DB/ROS
from ws_racr.execution.controller.robot_controller import (
    RobotController,
    _KNOWN_ROBOT_ACTIONS,
    _QUESTION_MARKERS,
)


def _make_controller() -> RobotController:
    """建立不連 DB/ROS 的 stub controller。"""
    ctrl = RobotController.__new__(RobotController)
    ctrl.memory_system = MagicMock()
    ctrl.memory_system.interaction_log = MagicMock()
    ctrl.task_manager = MagicMock()
    ctrl.voice_query_handler = None
    return ctrl


# ── _is_query_intent 新增模式 ─────────────────────────────────────────

def test_query_intent_muli_pattern_找():
    """'找六角扳手' 應被識別為查詢意圖（新增 找(.+) 模式）。"""
    ctrl = _make_controller()
    assert ctrl._is_query_intent("找六角扳手", action="pickup") is True


def test_query_intent_目前第幾步():
    """'目前在第幾步？' 應被識別為查詢意圖（補全 目前.*第幾步 模式）。"""
    ctrl = _make_controller()
    assert ctrl._is_query_intent("目前在第幾步？", action="stop") is True


def test_query_intent_還剩多少():
    """'還剩多少螺絲？' 應被識別為查詢意圖（新增 還剩多少 模式）。"""
    ctrl = _make_controller()
    assert ctrl._is_query_intent("還剩多少螺絲？", action="stop") is True


def test_query_intent_哪個方向():
    """'支架在平台的哪個方向？' 應被識別為查詢意圖（新增 哪個方向 模式）。"""
    ctrl = _make_controller()
    assert ctrl._is_query_intent("支架在平台的哪個方向？", action="stop") is True


def test_query_intent_第X步要什麼():
    """'第三步要什麼工具？' 應被識別為查詢意圖。"""
    ctrl = _make_controller()
    assert ctrl._is_query_intent("第三步要什麼工具？", action="stop") is True


# ── action-based 備援 ────────────────────────────────────────────────

def test_query_intent_action_fallback_unknown_action_with_question():
    """parser 未識別出已知 action + 含問句標誌詞 → 應視為查詢。"""
    ctrl = _make_controller()
    # 'unknown_action' 不在 _KNOWN_ROBOT_ACTIONS
    assert ctrl._is_query_intent("這個是什麼？", action="unknown_action") is True


def test_query_intent_action_fallback_known_action_no_false_positive():
    """已知 robot action + 不含問句標誌詞 → 不應誤判為查詢。"""
    ctrl = _make_controller()
    assert ctrl._is_query_intent("幫我拿六角扳手", action="pickup") is False


def test_query_intent_action_fallback_known_action_still_matches_pattern():
    """即使 action 是已知 robot action，只要 regex 匹配就視為查詢。"""
    ctrl = _make_controller()
    # parser 把 "找螺絲在哪裡" 解析成 pickup，但查詢模式應先命中
    assert ctrl._is_query_intent("螺絲在哪裡", action="pickup") is True


# ── _handle_query_via_voice_query 寫入 InteractionLog ───────────────

def test_handle_query_records_interaction():
    """_handle_query_via_voice_query() 成功後應呼叫 _record_interaction。"""
    ctrl = _make_controller()
    ctrl.voice_query_handler = MagicMock()
    ctrl.voice_query_handler.parse_and_execute.return_value = {
        'success': True,
        'query_type': 'inventory',
        'response_text': '桌上有六角扳手和圓柱。',
    }
    with patch.object(ctrl, '_record_interaction') as mock_record:
        result = ctrl._handle_query_via_voice_query("桌上有哪些物品？", task_id="t1")
    mock_record.assert_called_once()
    args = mock_record.call_args[0]
    assert args[1] == 'answer'
    assert '六角扳手' in args[2]
    assert result.get('policy_answer') is True
