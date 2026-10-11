"""Step definitions: the nine TSM step classes (``S_t``), their Chinese names and the allowed FSM transitions.

Single source of truth for step indices; the user says "step X" (1-8), the classifier index is X-1,
index 8 is ``idle``.

統一步驟定義 - 完全基於 TSM 視覺步驟辨識
整個系統統一使用 TSM Classifier 的 9 個步驟類別

TSM Classifier 定義的 9 個步驟（index 0-8）：
  0: place_platform (第1步)
  1: insert_cylinder_under_platform (第2步)
  2: tighten_small_screw_under_platform (第3步)
  3: place_bracket_on_table (第4步)
  4: align_pi_on_bracket (第5步)
  5: insert_upper_big_screw (第6步)
  6: align_bracket_to_platform (第7步)
  7: insert_lower_big_screw (第8步)
  8: idle (待機/未開始)

規則：用戶說「第X步」→ TSM index = X-1
     例如：「第三步」 = TSM index 2 = tighten_small_screw_under_platform
"""

# ========================================
# TSM 步驟定義（唯一真實來源）
# ========================================

# TSM Index → 步驟類別名稱
STEP_CLASSES = {
    0: 'place_platform',
    1: 'insert_cylinder_under_platform',
    2: 'tighten_small_screw_under_platform',
    3: 'place_bracket_on_table',
    4: 'align_pi_on_bracket',
    5: 'insert_upper_big_screw',
    6: 'align_bracket_to_platform',
    7: 'insert_lower_big_screw',
    8: 'idle',
}

# 步驟類別名稱 → TSM Index
CLASS_TO_INDEX = {v: k for k, v in STEP_CLASSES.items()}

# 步驟中文名稱（用於語音交互）
STEP_NAMES_ZH = {
    0: '第1步：放置平台到工作區',
    1: '第2步：插入4個圓柱到平台下方',
    2: '第3步：用扳手鎖緊4個小螺絲',
    3: '第4步：放置支架到桌上',
    4: '第5步：對齊PI工件到支架',
    5: '第6步：插入上方大螺絲',
    6: '第7步：對齊支架到平台',
    7: '第8步：插入下方大螺絲',
    8: '待機中',
}

# FSM 轉換規則（單向流程，禁止回退）
ALLOWED_TRANSITIONS = {
    None: ['idle', 'place_platform'],
    'idle': ['place_platform', 'idle'],
    'place_platform': ['insert_cylinder_under_platform', 'place_platform'],
    'insert_cylinder_under_platform': ['tighten_small_screw_under_platform', 'insert_cylinder_under_platform'],
    'tighten_small_screw_under_platform': ['place_bracket_on_table', 'tighten_small_screw_under_platform'],
    'place_bracket_on_table': ['align_pi_on_bracket', 'place_bracket_on_table'],
    'align_pi_on_bracket': ['insert_upper_big_screw', 'align_pi_on_bracket'],
    'insert_upper_big_screw': ['align_bracket_to_platform', 'insert_upper_big_screw'],
    'align_bracket_to_platform': ['insert_lower_big_screw', 'align_bracket_to_platform'],
    'insert_lower_big_screw': ['idle', 'insert_lower_big_screw'],
}


# ========================================
# 工具函數
# ========================================

def user_input_to_step_index(user_input: int) -> int:
    """用戶輸入的步驟編號轉換為 TSM index
    
    用戶說「第1步」 → 0
    用戶說「第2步」 → 1
    ...
    用戶說「第8步」 → 7
    用戶說「第9步」或 0 → 8 (idle)
    """
    if user_input == 0 or user_input == 9:
        return 8  # idle
    elif 1 <= user_input <= 8:
        return user_input - 1
    else:
        return 8  # 預設返回 idle


def step_index_to_user_display(step_index: int) -> int:
    """TSM index 轉換為用戶看到的步驟編號
    
    0 → 第1步
    1 → 第2步
    ...
    7 → 第8步
    8 → 0 (idle，特殊顯示)
    """
    if step_index == 8:
        return 0  # idle 顯示為步驟0
    elif 0 <= step_index <= 7:
        return step_index + 1
    else:
        return 0


def get_step_class(step_index: int) -> str:
    """獲取步驟的類別名稱"""
    return STEP_CLASSES.get(step_index, 'idle')


def get_step_index(class_name: str) -> int:
    """獲取類別名稱的 TSM index"""
    return CLASS_TO_INDEX.get(class_name, 8)


def get_step_name_zh(step_index: int) -> str:
    """獲取步驟的中文名稱"""
    return STEP_NAMES_ZH.get(step_index, '未知步驟')


def is_valid_transition(from_class: str, to_class: str) -> bool:
    """檢查步驟轉換是否合法"""
    allowed = ALLOWED_TRANSITIONS.get(from_class, [])
    return to_class in allowed
