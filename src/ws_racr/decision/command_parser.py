"""Command parser: turns one spoken sub-command into a structured skill JSON.

Cascade (the LLM is only a constrained fallback, never the default path):

* Stage A - normalisation: strip punctuation / STT noise
* Stage B - exact alias lookup against ``language_mapper`` vocabularies (0 ms)
* Stage C - fuzzy (difflib, optional pinyin) matching (<1 ms)
* Stage F - constrained LLM fallback, only when A-C fail, with a closed candidate list

Used by :class:`ws_racr.execution.controller.robot_controller.RobotController` after Layer 1 has
decided that a span is an EXECUTE intent.
"""
import logging
import json
import os
import re
import difflib
from typing import Dict, List, Optional, Tuple

try:
    from pypinyin import lazy_pinyin as _lazy_pinyin
    _PYPINYIN_AVAILABLE = True
except ImportError:
    _PYPINYIN_AVAILABLE = False

from ws_racr.config import get_openai_client, get_model_name
from ws_racr.decision.language_mapper import (
    ACTION_ALIASES, OBJECT_ALIASES, DIRECTION_ALIASES,
    ROTATION_ALIASES, SPATIAL_MODIFIER_ALIASES, SIZE_ALIASES,
    LOCATION_ALIASES,
)

logger = logging.getLogger(__name__)

# ── Phonetic helper ────────────────────────────────────────────────────────

def _to_pinyin(text: str) -> str:
    """中文 → 無空格拼音連字串，例如 '拿圓柱' → 'nayuanzhu'。pypinyin 未安裝時原樣返回。"""
    if not _PYPINYIN_AVAILABLE:
        return text
    return ''.join(_lazy_pinyin(text))


# ── Stage A: 正規化 ──────────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """清除標點、全形字元，把英文轉小寫，不改動中文字元。"""
    # 全形標點 → 移除
    for ch in '，。？！　：；「」『』【】《》':
        text = text.replace(ch, '')
    # 半形標點 → 空白
    text = re.sub(r'[,.\?!;:\[\](){}<>"\']', ' ', text)
    # 多空白 → 單空白
    text = ' '.join(text.split())
    # 英文小寫（中文不受影響）
    return text.lower().strip()


def _extract_core_command(text: str) -> str:
    """如果是帶 memory context 的增強輸入，只取 [Current Task] 後面的部分。"""
    if '[Current Task]' in text:
        return text.split('[Current Task]')[-1].strip()
    return text


# ── Stage B + C: Alias 匹配 ──────────────────────────────────────────────────

def _match_from_aliases(
    text: str,
    alias_map: Dict[str, List[str]],
    fuzzy_threshold: float = 0.60,
) -> Tuple[Optional[str], float, str]:
    """
    在 alias_map 中尋找 text 的最佳匹配。

    Stage B (exact): 從長到短掃所有 alias，若是 text 的 substring 直接返回。
    Stage C (fuzzy): 若無精確匹配，用 SequenceMatcher 找最高相似度。

    Returns:
        (canonical_key or None, confidence 0-1, source: 'exact'|'fuzzy'|'none')
    """
    # 建立 (alias, canonical) 列表並按 alias 長度降序排列（longest match first）
    candidates: List[Tuple[str, str]] = []
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            candidates.append((alias.lower(), canonical))
    candidates.sort(key=lambda x: len(x[0]), reverse=True)

    # Stage B: exact match（長度 >= 2，避免單字誤判，e.g. alias='原' 匹配 '原點'）
    # ASCII alias 用 word-boundary，防止 "pi" 誤匹配進 "pickup" 等英文動詞
    for alias, canonical in candidates:
        if len(alias) < 2:
            continue
        if alias.isascii():
            # 前後不能緊接英文字母，確保獨立詞彙
            if re.search(r'(?<![a-zA-Z])' + re.escape(alias) + r'(?![a-zA-Z])', text):
                return canonical, 1.0, 'exact'
        else:
            if alias in text:
                return canonical, 1.0, 'exact'

    # Stage C: fuzzy — 滑動視窗，每次只比對長度等於 alias 的片段
    # 解決問題：SequenceMatcher("圓柱", "拿圓柱給我") ≈ 0.4（被整句稀釋）
    # 改法 1：從 text 中切出所有長度 = len(alias) 的視窗，取最佳 char ratio
    # 改法 2：同時比對 pinyin 視窗（解決 STT 同音字錯誤，如 那原著 → 拿圓柱）
    text_py = _to_pinyin(text)
    best_canonical: Optional[str] = None
    best_score = 0.0
    for alias, canonical in candidates:
        n = len(alias)
        if n == 0:
            continue
        # --- char windowed ---
        best_window_ratio = 0.0
        if alias.isascii():
            # ASCII alias: require word-boundary to avoid sub-word false matches
            # (e.g. "pi" must NOT fuzzy-score 1.0 from window inside "pickup")
            if re.search(r'(?<![a-zA-Z])' + re.escape(alias) + r'(?![a-zA-Z])', text):
                best_window_ratio = 1.0
            # else stays 0.0 — partial sub-word matches don't contribute
        else:
            for i in range(max(1, len(text) - n + 1)):
                window = text[i:i + n]
                r = difflib.SequenceMatcher(None, alias, window).ratio()
                if r > best_window_ratio:
                    best_window_ratio = r
        # --- pinyin windowed ---
        # 真正的 STT 同音字錯誤（那原著→拿圓柱）pinyin ratio ≈ 1.0
        # 兩道防護：
        #   1. len(alias_py) >= 5：單字中文 pinyin 太短（na/fang=2-4），滑窗必中 → 排除
        #   2. best_py_ratio >= 0.82：低分 pinyin 不貢獻（false positive threshold）
        alias_py = _to_pinyin(alias)
        np_ = len(alias_py)
        best_py_ratio = 0.0
        if np_ >= 5:
            for i in range(max(1, len(text_py) - np_ + 1)):
                window_py = text_py[i:i + np_]
                rp = difflib.SequenceMatcher(None, alias_py, window_py).ratio()
                if rp > best_py_ratio:
                    best_py_ratio = rp
        pinyin_contribution = best_py_ratio if best_py_ratio >= 0.82 else 0.0
        ratio = max(best_window_ratio, pinyin_contribution)
        if ratio > best_score:
            best_score = ratio
            best_canonical = canonical

    if best_score >= fuzzy_threshold:
        return best_canonical, best_score, 'fuzzy'

    return None, 0.0, 'none'


# ── 數值提取輔助 ─────────────────────────────────────────────────────────────

def _chinese_num_to_int(s: str) -> Optional[int]:
    """把中文數字字串轉成整數。支援 '五'→5, '十二'→12, '二十五'→25, '一百'→100。"""
    digits = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
              '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    units = {'十': 10, '百': 100}
    result = 0
    current = 0
    for ch in s:
        if ch in digits:
            current = digits[ch]
        elif ch in units:
            unit = units[ch]
            result += (current or 1) * unit
            current = 0
        else:
            return None
    result += current
    return result if result > 0 else None


def _extract_distance(text: str, default: float = 50.0) -> float:
    """從文字中取距離數值（mm）。支援阿拉伯數字與中文數字。
    例如 '100mm' → 100.0, '五公分' → 50.0, '十二公分' → 120.0"""
    # 阿拉伯數字 + mm 單位
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:mm|毫米|公厘|公釐)', text)
    if m:
        val = float(m.group(1))
        if 1.0 <= val <= 500.0:
            return val
    # 阿拉伯數字 + 公分（× 10 → mm）
    m = re.search(r'(\d+(?:\.\d+)?)\s*公分', text)
    if m:
        val = float(m.group(1)) * 10.0
        if 1.0 <= val <= 2000.0:
            return val
    # 純阿拉伯數字（無單位，當 mm）
    m = re.search(r'(\d+(?:\.\d+)?)', text)
    if m:
        val = float(m.group(1))
        if 1.0 <= val <= 500.0:
            return val
    # 中文數字 + 公分（轉成 mm）
    m = re.search(r'([零一二三四五六七八九十百]+)\s*公分', text)
    if m:
        val = _chinese_num_to_int(m.group(1))
        if val is not None and 1 <= val <= 200:
            return float(val * 10)       # 公分 → mm
    # 中文數字 + 毫米/mm
    m = re.search(r'([零一二三四五六七八九十百]+)\s*(?:毫米|公厘|公釐)', text)
    if m:
        val = _chinese_num_to_int(m.group(1))
        if val is not None and 1 <= val <= 500:
            return float(val)
    return default


def _extract_rotation_angle(text: str, default: float = 45.0) -> float:
    """從文字中取旋轉角度（度）。例如 '45度' → 45.0"""
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:度|°|deg)', text)
    if m:
        return float(m.group(1))
    return default


def _extract_steps_back(text: str, default: int = 1) -> int:
    """從文字中取回退步數。例如 '三步之前' → 3"""
    chinese_digits = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                      '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
    m = re.search(r'([一二三四五六七八九十\d]+)\s*步', text)
    if m:
        raw = m.group(1)
        if raw in chinese_digits:
            return chinese_digits[raw]
        try:
            return int(raw)
        except ValueError:
            pass
    return default


# ── 主解析函式 ───────────────────────────────────────────────────────────────

def process_user_input(
    user_input: str,
    output_dir: str = "output",
    world_context: Optional[Dict] = None,
) -> Dict:
    """
    四層 Cascade 指令解析器。

    Stage A → Stage B → Stage C → (如仍無解) Stage F (受限 LLM)

    Args:
        user_input: 使用者的語音/文字指令
        output_dir: 輸出目錄（向後相容，目前未使用）
        world_context: 可選的世界狀態上下文，包含
            - visible_objects: List[str]  桌上可見物件
            - current_step: str           目前組裝步驟
            - robot_status: str           機器人狀態
            Stage F LLM fallback 會把這些資訊放入 prompt。

    Returns 與舊版相同的 dict schema，完全向後相容。
    """
    logger.info(f"解析指令: \"{user_input}\"")

    # ── Stage A: 正規化 ──
    core = _extract_core_command(user_input)   # 去掉 memory context 前綴
    text = _normalize_text(core)

    result: Dict = {
        "user_input": user_input,
        "success": False,
        "action": "unknown",
        "target_object": None,
        "object_size": None,
        "spatial_modifier": None,
        "target_location": None,
        "source_location": None,
        "direction": None,
        "distance": None,
        "rotation_direction": None,
        "rotation_angle": None,
        "parameters": {},
        "confidence": 0.0,
        "parse_source": "unresolved",
    }

    # ── Stage B + C: 各欄位獨立匹配 ──
    action,       action_conf,  action_src  = _match_from_aliases(text, ACTION_ALIASES)
    obj,          obj_conf,     _           = _match_from_aliases(text, OBJECT_ALIASES)
    direction,    dir_conf,     _           = _match_from_aliases(text, DIRECTION_ALIASES)
    rotation_dir, rot_conf,     _           = _match_from_aliases(text, ROTATION_ALIASES)
    spatial_mod,  _,            _           = _match_from_aliases(text, SPATIAL_MODIFIER_ALIASES)
    obj_size,     _,            _           = _match_from_aliases(text, SIZE_ALIASES)
    location,     _,            _           = _match_from_aliases(text, LOCATION_ALIASES)

    # ── 特殊規則：方向/旋轉 → 覆寫 action ──
    # 有方向詞 + action 是通用的移動 → 應為 move_relative
    if direction and dir_conf >= 0.8:
        if action in (None, 'move') or action_conf < 0.9:
            action, action_conf, action_src = 'move_relative', dir_conf, 'context'

    # 有旋轉方向 + action 不是 rotate_gripper → 覆寫
    if rotation_dir and rot_conf >= 0.8:
        if action not in ('rotate_gripper',) or action_conf < 0.9:
            action, action_conf, action_src = 'rotate_gripper', rot_conf, 'context'

    # ── 填入結果 ──
    if action and action != 'unknown':
        result.update({
            "success":          True,
            "action":           action,
            "target_object":    obj,
            "object_size":      obj_size,
            "spatial_modifier": spatial_mod,
            "target_location":  location,
            "direction":        direction,
            "rotation_direction": rotation_dir,
            "parse_source":     action_src,
        })

        if action == 'move_relative':
            result["distance"] = _extract_distance(text)

        if action == 'rotate_gripper':
            result["rotation_angle"] = _extract_rotation_angle(text)

        if action == 'return_to_last':
            result["parameters"]["steps_back"] = _extract_steps_back(text)

        # 整體信心度：action 權重 0.7，object 0.3（action 更關鍵）
        # 若無 object，直接用 action_conf（避免平均稀釋停止/返回等單槽指令）
        if obj:
            result["confidence"] = action_conf * 0.7 + obj_conf * 0.3
        else:
            result["confidence"] = action_conf

        print_result(result)
        return result

    # ── Stage F: 受限 LLM Fallback ──────────────────────────────────────────
    # Pass only the stripped core command — NOT the full enriched input with KG context,
    # to prevent the LLM from hallucinating objects mentioned in context headers.
    logger.warning(f" 規則解析失敗（無法識別動作），使用受限 LLM fallback...")
    return _fallback_llm_constrained(core, result, world_context=world_context)


def _fallback_llm_constrained(
    user_input: str,
    partial_result: Dict,
    world_context: Optional[Dict] = None,
) -> Dict:
    """
    Stage F: 只在規則/fuzzy 完全無法解析時才呼叫 LLM。
    傳入候選清單，避免 open-ended 輸出。
    world_context (optional):
        visible_objects: List[str]  — 桌上可見物件類別
        current_step:    str        — 目前組裝步驟
        robot_status:    str        — 機器人狀態 (idle / moving / grasping …)
    """
    try:
        client = get_openai_client()
        model_name = get_model_name()

        action_candidates = list(ACTION_ALIASES.keys())
        object_candidates = list(OBJECT_ALIASES.keys()) + ['null']
        location_candidates = list(LOCATION_ALIASES.keys()) + ['null']

        # ── 組裝世界狀態 context block ──
        context_lines: list = []
        if world_context:
            visible = world_context.get('visible_objects')
            if visible:
                vis_str = ', '.join(visible) if isinstance(visible, list) else str(visible)
                context_lines.append(f'桌上可見物件: {vis_str}')
            step = world_context.get('current_step')
            if step:
                context_lines.append(f'目前組裝步驟: {step}')
            robot_st = world_context.get('robot_status')
            if robot_st:
                context_lines.append(f'機器人狀態: {robot_st}')

        context_block = ''
        if context_lines:
            context_block = '\n目前環境狀態（可用來消除詞彙歧義）：\n' + '\n'.join(f'  {l}' for l in context_lines) + '\n'

        prompt = (
            f'語音辨識輸入（可能含雜訊）："{user_input}"\n'
            f'{context_block}\n'
            f'請從以下固定候選中選出最可能的解析，回答純 JSON：\n'
            f'action 候選：{action_candidates}\n'
            f'target_object 候選：{object_candidates}\n'
            f'target_location 候選：{location_candidates}\n\n'
            f'回答格式：{{"action":"...","target_object":"...","target_location":"...","confidence":0.0}}\n'
            f'若某欄位無對應，填 null。只回答 JSON，不要解釋。'
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=120,
        )

        content = response.choices[0].message.content.strip()
        # 容錯：去掉可能的 markdown code fence
        content = re.sub(r'^```[a-z]*\n?', '', content)
        content = re.sub(r'\n?```$', '', content).strip()
        parsed = json.loads(content)

        action = parsed.get("action", "unknown")
        if action not in action_candidates:
            action = "unknown"

        # 防止 LLM hallucination：object/location 必須在已知候選清單內
        raw_obj = parsed.get("target_object")
        obj = raw_obj if (raw_obj and raw_obj in object_candidates and raw_obj != 'null') else None

        raw_loc = parsed.get("target_location")
        loc = raw_loc if (raw_loc and raw_loc in location_candidates and raw_loc != 'null') else None

        partial_result.update({
            "success":         action != "unknown",
            "action":          action,
            "target_object":   obj,
            "target_location": loc,
            "confidence":      float(parsed.get("confidence", 0.5)),
            "parse_source":    "llm_fallback",
        })

        print_result(partial_result)
        return partial_result

    except Exception as e:
        logger.error(f"LLM fallback 失敗: {e}")
        partial_result["error"] = str(e)
        return partial_result

# 工具 schema（給 LLM 看的規格）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "parse_human_command",
            "description": "解析人類指令，轉換為機器人任務",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string", 
                        "enum": ["pickup", "grab", "grasp", "place", "put", "release", 
                                "move", "navigate", "go",
                                "handover", "give", "pass",
                                "gripper_open", "gripper_close", "stop", "emergency_stop",
                                "move_relative", "adjust_position", "nudge", "shift",
                                "return_to_last", "go_back", "return_to_origin", "go_home",
                                "rotate_gripper", "twist", "turn", "unknown"],
                        "description": "基本動作類型"
                    },
                    "rotation_direction": {
                        "type": "string",
                        "enum": ["left", "right", "clockwise", "counterclockwise", 
                                "左", "右", "順時針", "逆時針"],
                        "description": "旋轉方向（用於 rotate_gripper）"
                    },
                    "rotation_angle": {
                        "type": "number",
                        "description": "旋轉角度（度），預設為 45 度"
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right", "forward", "backward", "up", "down",
                                "左", "右", "前", "後", "上", "下"],
                        "description": "相對移動方向（用於 move_relative）"
                    },
                    "distance": {
                        "type": "number",
                        "description": "相對移動距離（mm），預設為 50mm"
                    },
                    "target_object": {
                        "type": "string", 
                        "description": "要操作的目標物件名稱"
                    },
                    "object_size": {
                        "type": "string",
                        "enum": ["big", "small", "large", "tiny", "大", "小", "大號", "小號"],
                        "description": "物件的尺寸描述（如果有提到）"
                    },
                    "spatial_modifier": {
                        "type": "string",
                        "enum": ["leftmost", "rightmost", "frontmost", "backmost", "nearest", "farthest", "center",
                                "最左邊", "最右邊", "最前面", "最後面", "最近", "最遠", "中間"],
                        "description": "空間位置描述詞，用於從多個物體中選擇特定位置的物體"
                    },
                    "target_location": {
                        "type": "string", 
                        "description": "目標位置或地點名稱"
                    },
                    "source_location": {
                        "type": "string", 
                        "description": "物件的來源位置"
                    },
                    "parameters": {
                        "type": "object", 
                        "description": "任務特定參數",
                        "properties": {
                            "movement_speed": {"type": "string", "description": "移動速度要求"},
                            "precision_level": {"type": "string", "description": "精確度要求"},
                            "wait_time": {"type": "number", "description": "等待時間（秒）"},
                            "steps_back": {"type": "integer", "description": "返回幾步之前的位置（用於 return_to_last），預設為 1"}
                        }
                    },
                    "confidence": {
                        "type": "number",
                        "description": "解析信心度 (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["action"]
            }
        }
    }
]

SYSTEM_PROMPT = """你是機器人指令解析助手。

任務：
- 解析人類的自然語言指令
- 識別基本動作類型和關鍵參數  
- 提供解析信心度評估

支援的基本動作：
* pickup/grab/grasp: 抓取物件
* place/put/release: 放置或釋放物件
* move/navigate/go: 移動到指定位置
* move_relative/adjust_position/nudge/shift: 相對移動（需搭配 direction 和 distance）
* return_to_last/go_back: 返回上一個位置（可搭配 parameters.steps_back 指定返回幾步前，預設 1）
* return_to_origin/go_home: 返回原點
* rotate_gripper/twist/turn: 旋轉夾爪（需搭配 rotation_direction 和 rotation_angle）
* handover/give/pass: 遞交給人類
* gripper_open: 開啟夾爪
* gripper_close: 關閉夾爪
* stop: 停止機器人移動
* emergency_stop: 緊急停止

範例：
- "抓取紅色方塊" → action: "pickup", target_object: "紅色方塊"
- "把最右邊的圓柱交給我" → action: "handover", target_object: "圓柱", spatial_modifier: "最右邊"
- "拿起最近的螺絲" → action: "pickup", target_object: "螺絲", spatial_modifier: "最近"
- "夾取大號螺絲" → action: "pickup", target_object: "螺絲", object_size: "大"
- "給我小螺絲" → action: "pickup", target_object: "螺絲", object_size: "小"
- "把杯子放到桌上" → action: "place", target_object: "杯子", target_location: "桌上"  
- "移動到初始位置" → action: "move", target_location: "初始位置"
- "move to origin" → action: "move", target_location: "origin"
- "move to standby" → action: "move", target_location: "standby"
- "go to storage area" → action: "move", target_location: "storage area"
- "往左移一點" → action: "move_relative", direction: "left", distance: 50
- "向上移動 100mm" → action: "move_relative", direction: "up", distance: 100
- "返回上個點" → action: "return_to_last", parameters: {"steps_back": 1}
- "回到三步之前的位置" → action: "return_to_last", parameters: {"steps_back": 3}
- "回到原點" → action: "return_to_origin"
- "往左轉 45 度" → action: "rotate_gripper", rotation_direction: "left", rotation_angle: 45
- "順時針轉" → action: "rotate_gripper", rotation_direction: "clockwise", rotation_angle: 45
- "開啟夾爪" → action: "gripper_open"
- "關閉夾爪" → action: "gripper_close"
- "停止移動" → action: "stop"
- "緊急停止" → action: "emergency_stop"

注意：
- "standby", "origin", "storage", "observation" 等是位置名稱，應該使用 action: "move" 配合 target_location

請準確識別動作類型和相關參數，並評估解析的信心度。
"""

def parse_llm_response(response, user_input: str) -> Dict:
    """解析 LLM 回應"""
    try:
        message = response.choices[0].message
        
        # 檢查是否有工具調用
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_call = message.tool_calls[0]
            if tool_call.function.name == "parse_human_command":
                function_args = json.loads(tool_call.function.arguments)
                
                result = {
                    "user_input": user_input,
                    "success": True,
                    "action": function_args.get("action", "unknown"),
                    "target_object": function_args.get("target_object"),
                    "object_size": function_args.get("object_size"),  # 新增：物件尺寸
                    "spatial_modifier": function_args.get("spatial_modifier"),  # 新增：空間描述詞
                    "target_location": function_args.get("target_location"),
                    "source_location": function_args.get("source_location"),
                    "parameters": function_args.get("parameters", {}),
                    "confidence": function_args.get("confidence", 0.8),
                    "raw_response": function_args
                }
                
                # ===== 智能修正：自動糾正不一致的 action =====
                # 修正 1: 如果有 direction 參數，action 應該是 move_relative
                if function_args.get("direction") and result["action"] in ["move", "navigate", "go"]:
                    logger.warning(f"自動修正: action '{result['action']}' + direction → 'move_relative'")
                    result["action"] = "move_relative"
                
                # 修正 2: 如果有 rotation_direction 參數，action 應該是 rotate_gripper
                if function_args.get("rotation_direction") and result["action"] not in ["rotate_gripper", "twist", "turn"]:
                    logger.warning(f"自動修正: action '{result['action']}' + rotation_direction → 'rotate_gripper'")
                    result["action"] = "rotate_gripper"
                
                # 修正 3: 如果是 return_to_origin/go_home，統一為 return_to_origin
                if result["action"] in ["go_home", "return_to_home"]:
                    result["action"] = "return_to_origin"
                
                # 修正 4: 標準化同義詞 action
                action_synonyms = {
                    "grab": "pickup",
                    "grasp": "pickup",
                    "put": "place",
                    "release": "place",
                    "give": "handover",
                    "pass": "handover",
                    "navigate": "move",
                    "go": "move",
                    "go_back": "return_to_last"
                }
                if result["action"] in action_synonyms:
                    original = result["action"]
                    result["action"] = action_synonyms[original]
                    logger.warning(f"自動修正: 同義詞 '{original}' → '{result['action']}'")
                
                # 特別處理 move_relative：提取 direction 和 distance
                if result["action"] in ["move_relative", "adjust_position", "nudge", "shift"]:
                    result["direction"] = function_args.get("direction")
                    result["distance"] = function_args.get("distance", 50.0)
                
                # 特別處理 rotate_gripper：提取 rotation_direction 和 rotation_angle
                if result["action"] in ["rotate_gripper", "twist", "turn"]:
                    result["rotation_direction"] = function_args.get("rotation_direction")
                    result["rotation_angle"] = function_args.get("rotation_angle", 45.0)
                
                return result
        
        # 如果沒有工具調用，返回基本回應
        return {
            "user_input": user_input,
            "success": False,
            "error": "LLM 沒有返回結構化結果",
            "action": "unknown",
            "confidence": 0.0,
            "raw_content": message.content if hasattr(message, 'content') else None
        }
        
    except Exception as e:
        return {
            "user_input": user_input,
            "success": False,
            "error": f"回應解析錯誤: {str(e)}",
            "action": "unknown",
            "confidence": 0.0
        }

def print_result(result: Dict):
    """輸出解析結果"""
    logger.info(f"\n解析結果:")
    
    if result["success"]:
        logger.info(f"  成功解析")
        logger.info(f"  動作: {result['action']}")
        
        if result.get('target_object'):
            logger.info(f"  目標物件: {result['target_object']}")
            
        if result.get('target_location'):
            logger.info(f"  目標位置: {result['target_location']}")
            
        if result.get('source_location'):
            logger.info(f"  來源位置: {result['source_location']}")
            
        if result.get('parameters'):
            logger.info(f"   參數: {result['parameters']}")
            
        logger.info(f"  信心度: {result['confidence']:.2f}")
        
        # 顯示完整的 JSON 輸出
        logger.info(f"\n完整 JSON 輸出:")
        import json
        logger.info(json.dumps(result, ensure_ascii=False, indent=2))
        
    else:
        logger.error(f"  解析失敗: {result['error']}")
        # 也顯示失敗時的 JSON
        logger.info(f"\n錯誤 JSON 輸出:")
        import json
        logger.info(json.dumps(result, ensure_ascii=False, indent=2))


def validate_parse_result(result: Dict) -> bool:
    """驗證解析結果是否有效"""
    if not result.get("success"):
        return False
    
    action = result.get("action")
    if not action or action == "unknown":
        return False
    
    confidence = result.get("confidence", 0.0)
    if confidence < 0.5:  # 信心度閾值
        return False
    
    return True

# 測試用例
SAMPLE_INPUTS = [
    "抓取紅色方塊",
    "把杯子放到桌上", 
    "移動到初始位置",
    "檢查物件狀態",
    "把這個遞給我",
    "等待 3 秒鐘",
    "慢慢移動到觀察位置",
]

def run_sample_tests():
    """執行範例測試"""
    logger.info("執行 LLM 解析測試")
    
    for i, user_input in enumerate(SAMPLE_INPUTS, 1):
        logger.info(f"\\n{'='*60}")
        logger.info(f"測試 {i}/{len(SAMPLE_INPUTS)}")
        logger.info(f"{'='*60}")
        
        result = process_user_input(user_input)
        
        if validate_parse_result(result):
            logger.info("解析結果有效")
        else:
            logger.warning("解析結果可能有問題")
        
def interactive_mode():
    """互動模式"""
    logger.info("LLM 指令解析互動模式")
    logger.info("輸入自然語言指令，輸入 'quit' 結束")
    
    while True:
        try:
            user_input = input("\\n請描述機器人動作: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                logger.info("結束互動模式")
                break
            
            if not user_input:
                continue
            
            result = process_user_input(user_input)
            
            if validate_parse_result(result):
                logger.info("\\n指令可以執行")
            else:
                logger.warning("\\n指令需要澄清或修正")
                
        except KeyboardInterrupt:
            logger.info("\\n結束互動模式")
            break
        except Exception as e:
            logger.error(f"錯誤: {e}")

def main():
    """主函數"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            interactive_mode()
        elif sys.argv[1] == "test":
            run_sample_tests()
        else:
            # 處理單一指令
            user_input = " ".join(sys.argv[1:])
            result = process_user_input(user_input)
    else:
        logger.info("使用方式:")
        logger.info("  python llm_parser.py interactive     # 互動模式")
        logger.info("  python llm_parser.py test           # 執行測試")
        logger.info("  python llm_parser.py \"抓取紅色方塊\"   # 單一指令")

if __name__ == "__main__":
    main()