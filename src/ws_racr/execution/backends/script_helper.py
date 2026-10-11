"""
TMScript 執行助手
封裝「發送 → 等待 → 驗證 → 更新 DB」的完整流程
"""
import logging
from typing import List, Dict, Optional
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class ScriptExecutionHelper:
    """
    TMScript 執行助手
    
    統一管理：
    1. 腳本發送
    2. 執行等待
    3. 結果驗證
    4. 數據庫更新
    """
    
    def __init__(self, communicator):
        """
        初始化
        
        Args:
            communicator: RobotCommunicator 實例
        """
        self.communicator = communicator
    
    def execute_and_track(
        self,
        scripts: List[str],
        task_id: Optional[str] = None,
        timeout: float = 60.0,
        description: str = ""
    ) -> Dict:
        """
        執行腳本並追蹤完整流程
        
        Args:
            scripts: TMScript 指令列表（不含 QueueTag）
            task_id: 任務 ID（用於更新數據庫）
            timeout: 超時時間（秒）
            description: 執行描述
        
        Returns:
            Dict: 執行報告
                {
                    "success": bool,
                    "command_count": int,
                    "execution_time": float,
                    "error": Optional[str],
                    "start_time": str,
                    "end_time": str,
                    "task_id": Optional[str]
                }
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"執行 TMScript 批次")
        if description:
            logger.info(f"描述: {description}")
        logger.info(f"指令數: {len(scripts)}")
        logger.info(f" 超時限制: {timeout}秒")
        if task_id:
            logger.info(f"任務 ID: {task_id}")
        logger.info(f"{'='*60}\n")
        
        start_time = datetime.now()
        
        # === 步驟 1: 發送 + 等待 + 驗證 ===
        result = self.communicator.execute_script_sequence(scripts, timeout=timeout)
        
        end_time = datetime.now()
        
        # 補充時間資訊
        result["start_time"] = start_time.isoformat()
        result["end_time"] = end_time.isoformat()
        result["task_id"] = task_id
        result["description"] = description
        
        # === 步驟 2: 打印報告 ===
        self._print_report(result)
        
        return result
    
    def _print_report(self, result: Dict):
        """
        打印執行報告
        
        Args:
            result: 執行結果
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"執行報告")
        logger.info(f"{'='*60}")
        
        if result["success"]:
            logger.info(f"狀態: 成功")
        else:
            logger.error(f"狀態: 失敗")
            logger.info(f"   錯誤: {result.get('error', '未知')}")
        
        logger.info(f"指令數: {result['command_count']}")
        logger.info(f" 執行時間: {result['execution_time']:.2f}秒")
        logger.info(f"開始: {result['start_time']}")
        logger.info(f"結束: {result['end_time']}")
        
        if result.get('task_id'):
            logger.info(f"任務 ID: {result['task_id']}")
        
        if result.get('description'):
            logger.info(f"描述: {result['description']}")
        
        logger.info(f"{'='*60}\n")
    
    def execute_single_script(
        self,
        script: str,
        task_id: Optional[str] = None,
        timeout: float = 30.0,
        description: str = ""
    ) -> Dict:
        """
        執行單條腳本（便捷方法）
        
        Args:
            script: 單條 TMScript 指令
            task_id: 任務 ID
            timeout: 超時時間
            description: 描述
        
        Returns:
            Dict: 執行報告
        """
        return self.execute_and_track(
            scripts=[script],
            task_id=task_id,
            timeout=timeout,
            description=description or f"執行單條指令: {script[:50]}..."
        )


# 測試範例
if __name__ == "__main__":
    # 模擬測試
    class MockCommunicator:
        def execute_script_sequence(self, scripts, timeout):
            print(f"Mock: 執行 {len(scripts)} 個指令")
            time.sleep(1.0)
            return {
                "success": True,
                "command_count": len(scripts),
                "execution_time": 1.0,
                "error": None
            }
    
    class MockTaskManager:
        def update_task_status(self, task_id, status, **kwargs):
            print(f"Mock: 更新任務 {task_id} → {status}")
    
    # 創建助手
    helper = ScriptExecutionHelper(
        communicator=MockCommunicator(),
        task_manager=MockTaskManager()
    )
    
    # 測試執行
    test_scripts = [
        "IO[EndModule].DO[0]=0",
        "PTP(\"CPP\",400,200,500,180,0,90,100,200,0,false)",
        "IO[EndModule].DO[0]=1"
    ]
    
    result = helper.execute_and_track(
        scripts=test_scripts,
        task_id="test_task_123",
        timeout=30.0,
        description="測試抓取任務"
    )
    
    print(f"\n最終結果: {result['success']}")
