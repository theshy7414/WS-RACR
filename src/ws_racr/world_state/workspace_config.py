"""
Workspace Configuration
定義工作區的預期物體清單和數量
"""

import logging
logger = logging.getLogger(__name__)


# 預期物體清單（類型 → 數量）
EXPECTED_OBJECTS = {
    # 結構件（單一）
    'platform': 1,
    'bracket': 1,
    'pi': 1,
    'hex_wrench': 1,
    
    # 圓柱（4個）
    'cylinder': 4,
    
    # 螺絲
    'screw_small': 4,  # 小螺絲 4 個
    'screw_big': 2,    # 大螺絲 2 個
    
    # 螺帽（2個）
    'nut': 2,
}

# 物體 ID 格式（用於驗證）
VALID_OBJECT_IDS = set()
for obj_type, count in EXPECTED_OBJECTS.items():
    for i in range(1, count + 1):
        VALID_OBJECT_IDS.add(f"{obj_type}_{i}")

# 物體 ID 列表（按類型分組）
OBJECT_IDS_BY_TYPE = {
    obj_type: [f"{obj_type}_{i}" for i in range(1, count + 1)]
    for obj_type, count in EXPECTED_OBJECTS.items()
}


def is_valid_object_id(object_id: str) -> bool:
    """檢查物體 ID 是否合法"""
    return object_id in VALID_OBJECT_IDS


def get_expected_count(object_type: str) -> int:
    """取得物體類型的預期數量"""
    return EXPECTED_OBJECTS.get(object_type, 0)


def get_valid_ids_for_type(object_type: str) -> list:
    """取得物體類型的所有合法 ID"""
    return OBJECT_IDS_BY_TYPE.get(object_type, [])


def validate_detections(detections: list) -> tuple:
    """
    驗證檢測結果
    
    Returns:
        (valid_detections, invalid_ids)
    """
    valid = []
    invalid = []
    
    for det in detections:
        obj_id = det.get('object_id')
        if obj_id and is_valid_object_id(obj_id):
            valid.append(det)
        else:
            invalid.append(obj_id)
    
    return valid, invalid


# 工作區物體總數
TOTAL_EXPECTED_OBJECTS = sum(EXPECTED_OBJECTS.values())

logger.info(f"Workspace Configuration Loaded:")
logger.info(f"   Expected Objects: {TOTAL_EXPECTED_OBJECTS}")
logger.info(f"   Valid IDs: {len(VALID_OBJECT_IDS)}")
