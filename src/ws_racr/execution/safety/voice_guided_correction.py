#!/usr/bin/env python3
"""
Voice Guided Correction
語音指導修正解析器

解析人類的修正指令：
- "往下5mm" → dz = -5
- "往左3mm" → dy = -3 (假設Y軸朝左)
- "沒夾到" → 通知失敗檢測器
- "夾取失敗" → 通知失敗檢測器
- "角度轉10度" → angle_offset = 10

支持組合指令：
- "往下5mm然後往左3mm"
- "再試一次，角度調10度"
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GraspOffset:
    """單次人類指導的抓取偏移量"""

    def __init__(self,
                 dx: float = 0.0,
                 dy: float = 0.0,
                 dz: float = 0.0,
                 angle_offset: float = 0.0,
                 approach_offset: float = 0.0):
        """
        Args:
            dx, dy, dz: 位置偏移量 (mm)
            angle_offset: 角度偏移 (度)
            approach_offset: 接近高度偏移 (mm)
        """
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.angle_offset = angle_offset
        self.approach_offset = approach_offset
        self.human_instruction = None  # 人類原始指令


class VoiceGuidedCorrection:
    """
    語音指導修正解析器
    
    將人類的自然語言指令轉換為可執行的調整參數
    """
    
    def __init__(self):
        """初始化解析器"""
        # 失敗通知關鍵詞
        self.failure_keywords = [
            '沒夾到', '沒抓到', '沒拿到', '沒夾住',
            '夾取失敗', '抓取失敗', '失敗了', '夾不到',
            '空的', '夾爪是空的', '沒東西', '沒有東西',
            '掉了', '滑掉了', '滑走了'
        ]
        
        # 方向關鍵詞映射
        self.direction_map = {
            # Z軸（上下）
            '下': ('dz', -1),
            '往下': ('dz', -1),
            '下面': ('dz', -1),
            '下方': ('dz', -1),
            '下移': ('dz', -1),
            '上': ('dz', +1),
            '往上': ('dz', +1),
            '上面': ('dz', +1),
            '上方': ('dz', +1),
            '上移': ('dz', +1),
            
            # Y軸（左右）
            '左': ('dy', -1),
            '往左': ('dy', -1),
            '左邊': ('dy', -1),
            '左移': ('dy', -1),
            '右': ('dy', +1),
            '往右': ('dy', +1),
            '右邊': ('dy', +1),
            '右移': ('dy', +1),
            
            # X軸（前後）
            '前': ('dx', +1),
            '往前': ('dx', +1),
            '前面': ('dx', +1),
            '前移': ('dx', +1),
            '後': ('dx', -1),
            '往後': ('dx', -1),
            '後面': ('dx', -1),
            '後退': ('dx', -1),
            '後移': ('dx', -1),
        }
        
        # 角度關鍵詞
        self.angle_keywords = ['角度', '轉', '旋轉', '傾斜', '偏轉']
        
        # 高度關鍵詞
        self.height_keywords = ['高', '低', '接近高度', '抬高', '降低']
        
        # 力度關鍵詞
        self.force_keywords = ['用力', '力', '力道', '夾緊', '放鬆', '輕']
        
        # 數字抽取正則
        self.number_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(mm|毫米|公分|cm|度)?')
    
    def parse(self, voice_command: str) -> Optional[GraspOffset]:
        """
        解析語音指令
        
        Args:
            voice_command: 人類語音指令
            
        Returns:
            GraspOffset or None
            若為失敗通知，返回 None（應使用 is_failure_notification 檢查）
        """
        if not voice_command:
            return None
        
        logger.info(f"解析語音指令: \"{voice_command}\"")
        
        # === 0. 檢查是否為失敗通知 ===
        if self.is_failure_notification(voice_command):
            logger.warning(f"   這是失敗通知，不是修正指令")
            return None
        
        adjustment = GraspOffset()
        adjustment.human_instruction = voice_command
        
        # === 1. 解析方向+距離 ===
        for direction, (axis, sign) in self.direction_map.items():
            if direction in voice_command:
                # 尋找數字
                numbers = self._extract_numbers(voice_command, start_pos=voice_command.index(direction))
                if numbers:
                    value = numbers[0] * sign
                    setattr(adjustment, axis, value)
                    logger.info(f"   {axis} = {value} mm")
        
        # === 2. 解析角度 ===
        for keyword in self.angle_keywords:
            if keyword in voice_command:
                numbers = self._extract_numbers(voice_command, start_pos=voice_command.index(keyword))
                if numbers:
                    adjustment.angle_offset = numbers[0]
                    logger.info(f"   angle_offset = {adjustment.angle_offset}°")
                    break
        
        # === 3. 解析高度調整 ===
        if any(kw in voice_command for kw in self.height_keywords):
            if '高' in voice_command or '抬' in voice_command:
                numbers = self._extract_numbers(voice_command)
                if numbers:
                    adjustment.approach_offset = numbers[0]
                else:
                    adjustment.approach_offset = 5.0  # 預設5mm
                logger.info(f"   approach_offset = +{adjustment.approach_offset} mm")
            elif '低' in voice_command or '降' in voice_command:
                numbers = self._extract_numbers(voice_command)
                if numbers:
                    adjustment.approach_offset = -numbers[0]
                else:
                    adjustment.approach_offset = -5.0  # 預設-5mm
                logger.info(f"   approach_offset = {adjustment.approach_offset} mm")
        
        # === 4. 檢查是否有有效調整 ===
        if (adjustment.dx != 0 or adjustment.dy != 0 or adjustment.dz != 0 or
            adjustment.angle_offset != 0 or adjustment.approach_offset != 0):
            return adjustment
        
        # === 5. 特殊指令處理 ===
        if self._is_retry_command(voice_command):
            logger.info("   → 重試（無調整）")
            return GraspOffset()  # 返回零調整
        
        logger.warning("   無法解析有效調整")
        return None
    
    def is_failure_notification(self, voice_command: str) -> bool:
        """
        判斷是否為失敗通知
        
        Args:
            voice_command: 語音指令
            
        Returns:
            True: 是失敗通知（如"沒夾到"）
            False: 是修正指令（如"往下5mm"）
        """
        return any(keyword in voice_command for keyword in self.failure_keywords)
    
    def extract_object_id_from_failure(self, voice_command: str) -> Optional[str]:
        """
        從失敗通知中提取物體 ID
        
        範例：
        - "沒夾到 cylinder_1" → "cylinder_1"
        - "cylinder 沒抓到" → "cylinder"
        - "夾取失敗" → None (無指定物體)
        
        Args:
            voice_command: 語音指令
            
        Returns:
            物體 ID or None
        """
        # 常見物體名稱
        object_names = ['cylinder', 'bracket', 'pi', 'nut', 'screw', 'platform', 'hex_wrench']
        
        for name in object_names:
            if name in voice_command.lower():
                # 檢查是否有數字後綴（如 cylinder_1）
                import re
                match = re.search(rf'{name}[_\s]*(\d+)', voice_command.lower())
                if match:
                    return f"{name}_{match.group(1)}"
                else:
                    return name
        
        return None
    
    def _extract_numbers(self, text: str, start_pos: int = 0) -> List[float]:
        """
        從文字中抽取數字
        
        Args:
            text: 輸入文字
            start_pos: 開始搜尋位置
            
        Returns:
            數字列表
        """
        numbers = []
        matches = self.number_pattern.finditer(text[start_pos:])
        
        for match in matches:
            value = float(match.group(1))
            unit = match.group(2)
            
            # 單位轉換
            if unit in ['公分', 'cm']:
                value *= 10  # 轉換為 mm
            
            numbers.append(value)
        
        return numbers
    
    def _is_retry_command(self, text: str) -> bool:
        """判斷是否為重試指令"""
        retry_keywords = ['再試', '重試', '再來', '再抓', '試試看', '再一次']
        return any(kw in text for kw in retry_keywords)
    
    def suggest_correction_examples(self) -> List[str]:
        """提供修正指令範例"""
        return [
            "往下5mm",
            "往左3mm",
            "往右2mm然後往下5mm",
            "角度轉10度",
            "再高一點",
            "再低一點",
            "往前10mm角度轉15度",
            "再試一次"
        ]


def get_voice_guided_correction() -> VoiceGuidedCorrection:
    """全局單例"""
    if not hasattr(get_voice_guided_correction, '_instance'):
        get_voice_guided_correction._instance = VoiceGuidedCorrection()
    return get_voice_guided_correction._instance
