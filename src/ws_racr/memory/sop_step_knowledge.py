"""
Shared SOP step knowledge (runtime-safe, non-LLM dependency).
"""

from typing import Optional

STEP_KNOWLEDGE: dict = {
    'place_platform': {
        'step': 1, 'zh': '放置平台到工作區',
        'required_objects': ['platform'],
        'robot_assistance': 'pickup platform，協助穩固定位',
    },
    'insert_cylinder_under_platform': {
        'step': 2, 'zh': '插入4個圓柱到平台下方',
        'required_objects': ['cylinder'],
        'robot_assistance': 'pickup cylinder，引導對齊孔位，依序完成4個',
    },
    'tighten_small_screw_under_platform': {
        'step': 3, 'zh': '用扳手鎖緊4個小螺絲',
        'required_objects': ['screw_small', 'hex_wrench'],
        'robot_assistance': 'pickup hex_wrench 交給人類，或穩定圓柱輔助鎖緊',
    },
    'place_bracket_on_table': {
        'step': 4, 'zh': '放置支架到桌上',
        'required_objects': ['bracket'],
        'robot_assistance': 'pickup bracket 並放置到指定位置',
    },
    'align_pi_on_bracket': {
        'step': 5, 'zh': '對齊PI工件到支架',
        'required_objects': ['pi', 'bracket'],
        'robot_assistance': 'pickup pi，引導對齊支架孔位',
    },
    'insert_upper_big_screw': {
        'step': 6, 'zh': '插入上方大螺絲',
        'required_objects': ['screw_big'],
        'robot_assistance': 'handover screw_big，或固定組件輔助插入',
    },
    'align_bracket_to_platform': {
        'step': 7, 'zh': '對齊支架到平台',
        'required_objects': ['bracket', 'platform'],
        'robot_assistance': '穩定平台基座，協助對齊孔位',
    },
    'insert_lower_big_screw': {
        'step': 8, 'zh': '插入下方大螺絲並鎖螺帽',
        'required_objects': ['screw_big', 'nut'],
        'robot_assistance': 'handover screw_big 或 nut，固定螺絲防止旋轉',
    },
    'idle': {
        'step': None, 'zh': '待機中',
        'required_objects': [],
        'robot_assistance': '等待使用者指令',
    },
}


def get_step_knowledge(step_class: str) -> Optional[dict]:
    return STEP_KNOWLEDGE.get(step_class)

