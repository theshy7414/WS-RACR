"""
語言映射器 - 處理中英文工件名稱和動作指令的轉換
Language Mapper - Handles Chinese-English mapping for objects and actions
"""
import logging
from typing import Dict, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)

# ── Module-level alias tables (canonical source for command parser) ──────────
# Key = canonical value passed downstream; list = Chinese/STT variants.
# Lists are intentionally ordered longest-first so greedy substring matching
# picks the most specific alias before falling back to a single character.

ACTION_ALIASES: Dict[str, List[str]] = {
    'pickup':           ['拿起', '抓取', '夾取', '拿取', '甲取', '甲曲', '拿', '抓', '夾', '甲', '加', '取',
                         'pickup', 'grab', 'grasp'],
    'place':            ['放下', '放置', '放', '擺', 'place', 'put'],
    'handover':         ['交給', '遞給', '拿給', '給我', '給', 'handover', 'give', 'pass'],
    'move':             ['移動', '過去', '過來', '靠近', 'move', 'navigate'],
    'move_relative':    ['往左移', '往右移', '往上移', '往下移', '往前移', '往後移',
                         '向左移', '向右移', '向上移', '向下移', '向前移', '向後移',
                         'move_relative', 'adjust_position', 'nudge'],
    'return_to_origin': ['回到原點', '回原點', '回到home', '回home', '回到origin', '回origin',
                         '歸位', '回家', 'return_to_origin', 'go_home'],
    'return_to_last':   ['回上一個點', '返回上個點', '回上個點', '上一個位置',
                         'return_to_last', 'go_back'],
    'rotate_gripper':   ['旋轉', '扭', '轉', 'rotate_gripper', 'twist', 'turn'],
    'gripper_open':     ['放開夾爪', '打開夾爪', '張開夾爪', '鬆開夾爪', '放開夾', '放開', '打開', '張開', '鬆開',
                         'gripper_open', 'open_gripper'],
    'gripper_close':    ['關閉夾爪', '夾緊夾爪', '夾緊', '關閉', 'gripper_close', 'close_gripper'],
    'prepare':          ['準備好', '備好', '準備', 'prepare'],
    'stop':             ['停止', '暫停', '停', 'stop', 'halt'],
    'emergency_stop':   ['緊急停止', '立刻停', 'emergency_stop'],
    'retreat':          ['後退', '退後', '離開', 'retreat', 'back_off'],
}

OBJECT_ALIASES: Dict[str, List[str]] = {
    # Key = YOLO class name used by vision system
    # Each entry includes both Chinese aliases (for voice STT) and English aliases
    # (for LLM-generated sub-commands that use canonical class names).
    'cylinder':    ['圓柱體', '圓柱', '圓筒', '柱子', 'cylinder'],
    'platform':    ['平台', '平臺', '底座', 'platform'],
    'bracket':     ['支架', '架子', 'bracket'],
    'pi':          ['PI字型工件', 'PI工件', 'pi字型', 'pi工件', 'PI', 'Pi', 'pi'],
    'hex_wrench':  ['六角扳手', '六角扳', '扳手', '六角', 'hex_wrench', 'hex wrench', 'wrench'],
    'screw_big':   ['上方大螺絲', '下方大螺絲', '大螺絲', 'screw_big', 'big screw'],
    'screw_small': ['細螺絲', '小螺絲', 'screw_small', 'small screw'],
    'screw':       ['螺絲'],
    'nut':         ['螺帽', '螺母', 'nut'],
}

DIRECTION_ALIASES: Dict[str, List[str]] = {
    'left':     ['向左', '往左', '左邊', 'left'],
    'right':    ['向右', '往右', '右邊', 'right'],
    'forward':  ['向前', '往前', '前面', 'forward'],
    'backward': ['向後', '往後', '後面', 'backward', 'back'],
    'up':       ['向上', '往上', '上面', 'up'],
    'down':     ['向下', '往下', '下面', 'down'],
    # NOTE: bare single chars ('下','上','左','右','前','後') intentionally excluded
    # to prevent ambiguous matches (e.g. '一下' → down, '之後' → backward, '桌上' → up)
}

ROTATION_ALIASES: Dict[str, List[str]] = {
    'clockwise':        ['順時針', '右旋', 'clockwise'],
    'counterclockwise': ['逆時針', '左旋', 'counterclockwise'],
    'left':             ['往左轉', '左轉'],
    'right':            ['往右轉', '右轉'],
}

SPATIAL_MODIFIER_ALIASES: Dict[str, List[str]] = {
    'leftmost':  ['最左邊', '最左'],
    'rightmost': ['最右邊', '最右'],
    'frontmost': ['最前面', '最前'],
    'backmost':  ['最後面', '最後'],
    'nearest':   ['最近的', '最近', '近的'],
    'farthest':  ['最遠的', '最遠', '遠的'],
    'center':    ['中央', '中間'],
}

SIZE_ALIASES: Dict[str, List[str]] = {
    'big':   ['大號', '大的', '大', 'big', 'large'],
    'small': ['小號', '小的', '小', 'small', 'tiny'],
}

# Known canonical location keys → Chinese/English variants
# These keys must match keys in RobotController.known_locations
LOCATION_ALIASES: Dict[str, List[str]] = {
    'standby':     ['待機位置', '初始位置', '待機點', '待機', 'standby', 'standby position',
                    'initial', 'initial position', 'home position'],
    'origin':      ['原點位置', '原點', '原位', 'origin', 'home'],
    'storage':     ['收納位置', '儲存位置', '放置區', '收納', '存放',
                    'storage', 'storage area', 'storage position'],
    'observation': ['觀察位置', '俯視位置', '觀察', 'observation', 'observation area',
                    'observation position'],
}


class LanguageMapper:
    """中英文語言映射器"""
    
    def __init__(self):
        """初始化映射表"""
        
        # 工件名稱映射 (中文 → 英文)
        self.object_zh_to_en = {
            # 工具
            '螺絲起子': 'screwdriver',
            '起子': 'screwdriver',
            '六角扳手': 'hex_wrench',
            '扳手': 'wrench',
            '六角': 'hex_wrench',
            
            # 零件
            '平台': 'platform',
            '底座': 'platform',
            '支架': 'bracket',
            'PI字型工件': 'pi_shaped_part',
            'PI工件': 'pi_shaped_part',
            'pi字型': 'pi_shaped_part',
            'pi': 'pi_shaped_part',
            
            # 螺絲類
            '螺絲': 'screw',
            '大螺絲': 'screw_big',
            '小螺絲': 'screw_small',
            '上方大螺絲': 'screw_big',
            '下方大螺絲': 'screw_big',
            '圓柱': 'cylinder',
            
            # 其他
            '物體': 'object',
            '物件': 'object',
            '東西': 'object',
            '那個': 'object',
        }
        
        # 工件名稱映射 (英文 → 中文)
        self.object_en_to_zh = {v: k for k, v in self.object_zh_to_en.items()}
        
        # 添加額外的英文變體
        self.object_en_to_zh.update({
            'tool': '工具',
            'part': '零件',
            'component': '零件',
        })
        
        # 動作映射 (中文 → 英文)
        self.action_zh_to_en = {
            # 抓取相關
            '拿': 'pickup',
            '抓': 'pickup',
            '夾': 'pickup',
            '甲': 'pickup',  # 語音識別常見錯誤："夾" → "甲"
            '加': 'pickup',  # 語音識別常見錯誤："夾" → "加"
            '拿起': 'pickup',
            '抓取': 'pickup',
            '取': 'pickup',
            '拿取': 'pickup',
            '夾取': 'pickup',
            '甲取': 'pickup',  # 語音識別錯誤
            '甲曲': 'pickup',  # 語音識別錯誤："夾取" → "甲曲"
            
            # 放置相關
            '放': 'place',
            '放下': 'place',
            '放置': 'place',
            '擺': 'place',
            
            # 移動相關
            '移動': 'move',
            '移': 'move',
            '去': 'move',
            '到': 'move',
            '過去': 'move',
            '過來': 'move',
            '靠近': 'move',
            
            # 交接相關
            '給': 'handover',
            '交給': 'handover',
            '遞給': 'handover',
            '拿給': 'handover',
            
            # 夾爪相關
            '放開夾爪': 'gripper_open',
            '放開夾': 'gripper_open',
            '打開夾爪': 'gripper_open',
            '打開': 'gripper_open',
            '張開': 'gripper_open',
            '鬆開': 'gripper_open',
            '放開': 'gripper_open',
            '關閉夾爪': 'gripper_close',
            '關閉': 'gripper_close',
            '夾緊': 'gripper_close',
            
            # 旋轉相關
            '轉': 'rotate_gripper',
            '旋轉': 'rotate_gripper',
            '扭': 'rotate_gripper',
            
            # 準備相關
            '準備': 'prepare',
            '備好': 'prepare',
            
            # 停止相關
            '停': 'stop',
            '停止': 'stop',
            '暫停': 'stop',
            
            # 後退相關
            '後退': 'retreat',
            '退後': 'retreat',
            '離開': 'retreat',
        }
        
        # 動作映射 (英文 → 中文)
        self.action_en_to_zh = {
            'pickup': '拿取',
            'grasp': '抓取',
            'grab': '抓',
            'place': '放置',
            'put': '放',
            'release': '釋放',
            'move': '移動',
            'navigate': '移動',
            'go': '前往',
            'handover': '交給',
            'transfer': '轉交',
            'gripper_open': '打開夾爪',
            'gripper_close': '關閉夾爪',
            'rotate_gripper': '旋轉夾爪',
            'prepare': '準備',
            'stop': '停止',
            'halt': '停止',
            'retreat': '後退',
        }
        
        # 位置映射
        self.location_zh_to_en = {
            '這裡': 'here',
            '那裡': 'there',
            '前面': 'front',
            '後面': 'back',
            '旁邊': 'side',
            '上面': 'above',
            '下面': 'below',
            '工作區': 'workspace',
            '桌上': 'table',
            '待機位置': 'standby',
            '原點': 'origin',
            '人手': 'hand',
            '手上': 'hand',
        }
        
        self.location_en_to_zh = {v: k for k, v in self.location_zh_to_en.items()}
    
    def parse_command(self, command: str) -> Dict:
        """
        解析中文指令，提取動作、物體和參數
        
        Args:
            command: 中文指令，例如 "把螺絲起子拿給我"
            
        Returns:
            Dict: {
                'action': str,  # 動作 (英文)
                'objects': List[str],  # 物體列表 (英文)
                'locations': List[str],  # 位置列表 (英文)
                'original': str,  # 原始指令
                'action_zh': str,  # 動作 (中文)
                'objects_zh': List[str],  # 物體列表 (中文)
            }
        """
        result = {
            'action': None,
            'action_zh': None,
            'objects': [],
            'objects_zh': [],
            'locations': [],
            'locations_zh': [],
            'original': command
        }
        
        # 1. 提取動作（長詞優先，防止短詞如 '夾' 蓋住 '放開夾''）
        for zh, en in sorted(self.action_zh_to_en.items(), key=lambda x: -len(x[0])):
            if zh in command:
                result['action'] = en
                result['action_zh'] = zh
                break
        
        # 如果沒找到動作，使用默認
        if not result['action']:
            # 嘗試根據關鍵詞推斷
            if '給我' in command or '遞給' in command:
                result['action'] = 'handover'
                result['action_zh'] = '交給'
            elif any(word in command for word in ['拿', '取', '抓']):
                result['action'] = 'pickup'
                result['action_zh'] = '拿取'
            else:
                result['action'] = 'move'
                result['action_zh'] = '移動'
        
        # 2. 提取物體
        for zh, en in self.object_zh_to_en.items():
            if zh in command:
                result['objects'].append(en)
                result['objects_zh'].append(zh)
        
        # 3. 提取位置
        for zh, en in self.location_zh_to_en.items():
            if zh in command:
                result['locations'].append(en)
                result['locations_zh'].append(zh)
        
        return result
    
    def translate_to_task_json(self, command: str) -> Dict:
        """
        將中文指令轉換為 task_json 格式
        
        Args:
            command: 中文指令
            
        Returns:
            task_json: {
                'action': str,
                'target_object': str,
                'destination': str,
                'parameters': Dict
            }
        """
        parsed = self.parse_command(command)
        
        task_json = {
            'action': parsed['action'],
            'parameters': {
                'original_command': command,
                'language': 'zh'
            }
        }
        
        # 添加目標物體
        if parsed['objects']:
            task_json['target_object'] = parsed['objects'][0]
            task_json['parameters']['target_object_zh'] = parsed['objects_zh'][0]
        
        # 添加目標位置
        if parsed['locations']:
            task_json['destination'] = parsed['locations'][0]
            task_json['parameters']['destination_zh'] = parsed['locations_zh'][0]
        
        # 特殊處理："給我" → handover to human
        if '給我' in command or '遞給我' in command:
            task_json['destination'] = 'hand'
            task_json['parameters']['handover_to_human'] = True
        
        return task_json
    
    def translate_object_name(self, name: str, to_language: str = 'en') -> Optional[str]:
        """
        翻譯物體名稱
        
        Args:
            name: 物體名稱
            to_language: 目標語言 ('en' 或 'zh')
            
        Returns:
            翻譯後的名稱，如果找不到則返回原名稱
        """
        if to_language == 'en':
            return self.object_zh_to_en.get(name, name)
        else:
            return self.object_en_to_zh.get(name, name)
    
    def get_possible_names(self, canonical_name: str) -> List[str]:
        """
        獲取物體的所有可能名稱（中英文變體）
        
        Args:
            canonical_name: 標準名稱
            
        Returns:
            可能的名稱列表
        """
        names = [canonical_name]
        
        # 添加英文翻譯
        if canonical_name in self.object_zh_to_en:
            names.append(self.object_zh_to_en[canonical_name])
        
        # 添加中文翻譯
        if canonical_name in self.object_en_to_zh:
            names.append(self.object_en_to_zh[canonical_name])
        
        # 添加所有映射到同一英文名的中文名
        if canonical_name in self.object_zh_to_en:
            en_name = self.object_zh_to_en[canonical_name]
            for zh, en in self.object_zh_to_en.items():
                if en == en_name and zh != canonical_name:
                    names.append(zh)
        
        return names


# 全局實例
language_mapper = LanguageMapper()


def test_language_mapper():
    """測試語言映射器"""
    mapper = LanguageMapper()
    
    logger.info("=== 語言映射器測試 ===\n")
    
    test_commands = [
        "把螺絲起子拿給我",
        "拿起平台",
        "移動到待機位置",
        "準備六角扳手",
        "放下支架",
        "打開夾爪",
        "把那個東西給我",
    ]
    
    for cmd in test_commands:
        logger.info(f"指令: {cmd}")
        parsed = mapper.parse_command(cmd)
        logger.info(f"  解析結果: {parsed}")
        
        task_json = mapper.translate_to_task_json(cmd)
        logger.info(f"  Task JSON: {task_json}")


if __name__ == "__main__":
    test_language_mapper()
