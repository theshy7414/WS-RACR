"""
TTS 語音合成處理器 - 支援中文

架構：
  - 單一 worker thread 消費 PriorityQueue，消除 pygame.mixer 競爭條件
  - 啟動時預合成常用短句快取到 bytes，後續呼叫幾乎即時 (<50ms)
  - 動態內容（LLM 生成回答）走 edge-tts 雲端合成（約 500ms-1.5s）
  - priority 越小越優先；interrupt=True 清空佇列並插隊
  - wait_until_idle()：阻塞至已 speak 的句子全部播完（供 Orchestrator「播完再動」）

Priority 慣例：
  0  — 安全 / 緊急（interrupt=True 搭配使用）
  1  — 手勢確認（需要快速回饋）
  3  — 任務進度 / step 宣告
  5  — 一般確認 / 完成訊息（預設）
  7  — LLM 生成的長回答
"""

import logging
import os
import queue
import threading
import tempfile
import hashlib
import io
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 預快取短句清單（啟動時自動合成）
# ---------------------------------------------------------------------------
PRELOAD_PHRASES = [
    # 安全
    "緊急停止！",
    "偵測到安全風險，後退中",
    # 手勢
    "好的，我來了",
    "我在這裡，需要我幫忙嗎？",
    "我看不清楚你在指哪裡",
    # 任務結果
    "完成了",
    "好的，我再試一次",
    "好的，取消重試",
    # 障礙物
    "抱歉，我找不到障礙物",
    "抱歉，我無法抓取障礙物",
    "抱歉，我無法放置障礙物",
    "障礙物已移除，現在重新執行原來的任務",
    "抱歉，移除障礙物時發生錯誤",
    "抱歉，任務還是無法完成",
    # Recovery
    "Recovery 不確定，請提供驗證資訊",
    # 遞送
    "所有零件已遞送完畢",
    "零件遞送失敗",
    # 系統
    "系統待機中",
]


class TTSHandler:
    """
    TTS 處理器 v2 — worker queue + 預快取 + interrupt 支援

    修正問題：
      - TTS-C1: pygame.mixer 單例 race condition（多執行緒同時呼叫 load/play）
        → 改為單一 worker thread 序列化所有 TTS 播放
      - 短句高延遲（每次都走 edge-tts HTTP）
        → 啟動時預合成快取，後續呼叫直接讀 bytes
    """

    def __init__(self, voice: str = 'zh-TW-HsiaoChenNeural', rate: str = '+10%'):
        self.voice = voice
        self.rate = rate
        self.backend = self._init_backend()

        # PriorityQueue：(priority, seq, text)；seq 保證同優先級 FIFO
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = 0
        self._seq_lock = threading.Lock()

        # 預合成快取：text → MP3 bytes
        self._cache: dict[str, bytes] = {}
        self._cache_lock = threading.RLock()

        # Worker thread（序列化所有播放，消除 pygame race）
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        # 供 wait_until_idle：worker 正在合成／播放時為 True
        self._playing_lock = threading.Lock()
        self._playing = False
        # speak() 次數 − worker 已播完次數；避免兩句之間 queue 瞬間空誤判 idle
        self._pending_lock = threading.Lock()
        self._pending = 0

        # 啟動時背景預合成常用短句
        if self.backend == 'edge-tts':
            threading.Thread(target=self._preload_phrases, daemon=True).start()

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def speak(self, text: str, priority: int = 5, interrupt: bool = False):
        """
        播報語音。

        Args:
            text:      要播報的文字
            priority:  0=最高（安全）… 7=最低（LLM 長回答）；預設 5
            interrupt: True 時清空現有佇列並插隊（用於安全/緊急訊息）
        """
        if not text or not text.strip():
            return
        if self.backend == 'none':
            logger.info(f"TTS: {text}")
            return

        if interrupt:
            self._drain_queue()

        with self._seq_lock:
            self._seq += 1
            seq = self._seq

        with self._pending_lock:
            self._pending += 1
        try:
            self._queue.put((priority, seq, text))
        except Exception:
            with self._pending_lock:
                self._pending -= 1
            raise

    def wait_until_idle(self, timeout: float = 120.0) -> bool:
        """
        阻塞直到所有已 speak() 的句子都播完（_pending==0）且 worker 未在播／合成。

        用於「播完再動」：Orchestrator 在 _tool_execute 前呼叫。
        backend 為 none 時視為已空閒。
        """
        if self.backend == "none":
            return True
        import time

        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            with self._pending_lock:
                pend = self._pending
            with self._playing_lock:
                busy = self._playing
            if pend == 0 and not busy:
                return True
            time.sleep(0.03)
        return False

    def stop(self):
        """清空播放佇列（不停止正在播的音訊）。"""
        self._drain_queue()

    # ------------------------------------------------------------------
    # 內部：worker loop
    # ------------------------------------------------------------------

    def _worker_loop(self):
        """單一 worker thread — 序列消費佇列，避免 pygame 競爭。"""
        while True:
            try:
                _, _, text = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            with self._playing_lock:
                self._playing = True
            try:
                try:
                    with self._cache_lock:
                        cached = self._cache.get(text)

                    if cached is not None:
                        self._play_bytes(cached)
                    elif self.backend == 'edge-tts':
                        mp3_bytes = self._synthesize_edge_tts(text)
                        if mp3_bytes:
                            with self._cache_lock:
                                self._cache[text] = mp3_bytes  # 動態生成也快取
                            self._play_bytes(mp3_bytes)
                    elif self.backend == 'gtts':
                        self._play_gtts(text)
                    elif self.backend == 'pyttsx3':
                        self._play_pyttsx3(text)
                except Exception as e:
                    logger.warning(f"TTS 播報失敗: {e} | text={text!r}")
            finally:
                with self._playing_lock:
                    self._playing = False
                with self._pending_lock:
                    self._pending -= 1

    # ------------------------------------------------------------------
    # 內部：合成與播放
    # ------------------------------------------------------------------

    def _synthesize_edge_tts(self, text: str) -> Optional[bytes]:
        """合成 edge-tts → 回傳 MP3 bytes（網路呼叫，約 500ms-1.5s）。"""
        try:
            import edge_tts
            import asyncio

            async def _async():
                communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
                buf = io.BytesIO()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        buf.write(chunk["data"])
                return buf.getvalue()

            return asyncio.run(_async())
        except Exception as e:
            logger.warning(f"edge-tts 合成失敗: {e}")
            return None

    def _play_bytes(self, mp3_bytes: bytes):
        """使用 pygame 播放 MP3 bytes（需在 worker thread 執行）。"""
        import pygame

        if not pygame.mixer.get_init():
            pygame.mixer.init()

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(mp3_bytes)
            tmp_path = f.name

        try:
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(20)
        finally:
            pygame.mixer.music.unload()
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _play_gtts(self, text: str):
        """gTTS fallback。"""
        from gtts import gTTS
        import pygame

        if not pygame.mixer.get_init():
            pygame.mixer.init()

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            tmp_path = f.name
        try:
            gTTS(text=text, lang='zh-tw', slow=False).save(tmp_path)
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(20)
        finally:
            pygame.mixer.music.unload()
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _play_pyttsx3(self, text: str):
        """pyttsx3 fallback（離線，中文支援有限）。"""
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        for v in engine.getProperty('voices'):
            if 'chinese' in v.name.lower() or 'mandarin' in v.name.lower():
                engine.setProperty('voice', v.id)
                break
        engine.say(text)
        engine.runAndWait()

    # ------------------------------------------------------------------
    # 內部：預快取 + 佇列管理
    # ------------------------------------------------------------------

    def _preload_phrases(self):
        """啟動時背景預合成 PRELOAD_PHRASES，合成完存入 _cache。"""
        logger.info(f"TTS 預快取：合成 {len(PRELOAD_PHRASES)} 個常用短句...")
        ok = 0
        for phrase in PRELOAD_PHRASES:
            try:
                mp3 = self._synthesize_edge_tts(phrase)
                if mp3:
                    with self._cache_lock:
                        self._cache[phrase] = mp3
                    ok += 1
            except Exception as e:
                logger.warning(f"預快取失敗: {phrase!r} — {e}")
        logger.info(f"TTS 預快取完成：{ok}/{len(PRELOAD_PHRASES)} 成功")

    def _drain_queue(self):
        """清空佇列中尚未播放的項目（interrupt 時使用）；同步扣回 speak() 已計入的 _pending。"""
        drained = 0
        while True:
            try:
                self._queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            with self._pending_lock:
                self._pending = max(0, self._pending - drained)
            logger.info(f"   TTS 佇列清空 ({drained} 項被丟棄)")

    # ------------------------------------------------------------------
    # 後端偵測
    # ------------------------------------------------------------------

    def _init_backend(self) -> str:
        try:
            import edge_tts  # noqa: F401
            logger.info("TTS Backend: edge-tts (Microsoft Azure)")
            return 'edge-tts'
        except ImportError:
            pass
        try:
            from gtts import gTTS  # noqa: F401
            logger.info("TTS Backend: gTTS (Google)")
            return 'gtts'
        except ImportError:
            pass
        try:
            import pyttsx3  # noqa: F401
            logger.warning("TTS Backend: pyttsx3 (離線，中文支援有限)")
            return 'pyttsx3'
        except ImportError:
            pass
        logger.error("無可用的 TTS 引擎")
        return 'none'


# ---------------------------------------------------------------------------
# Priority 常數（呼叫端使用）
# ---------------------------------------------------------------------------
TTS_PRIORITY_SAFETY  = 0   # 緊急停止、安全後退
TTS_PRIORITY_GESTURE = 1   # 手勢確認（需快速回饋）
TTS_PRIORITY_TASK    = 3   # 任務進度、step 宣告
TTS_PRIORITY_NORMAL  = 5   # 一般確認（預設）
TTS_PRIORITY_LLM     = 7   # LLM 生成的長回答


# ---------------------------------------------------------------------------
# 測試
# ---------------------------------------------------------------------------

def test_tts():
    logger.info("\n" + "=" * 70)
    logger.info("測試 TTSHandler v2（queue + cache）")
    logger.info("=" * 70)

    tts = TTSHandler()

    import time
    logger.info("\n等待預快取完成（最多 10s）...")
    time.sleep(10)

    logger.info("\n[1] 測試預快取短句（應幾乎即時）")
    t0 = time.time()
    tts.speak("好的，我來了", priority=TTS_PRIORITY_GESTURE)
    time.sleep(3)
    logger.info(f"   排入佇列耗時: {(time.time()-t0)*1000:.0f}ms")

    logger.info("\n[2] 測試連續兩句（不應互相覆蓋）")
    tts.speak("完成了", priority=TTS_PRIORITY_NORMAL)
    tts.speak("系統待機中", priority=TTS_PRIORITY_NORMAL)
    time.sleep(5)

    logger.info("\n[3] 測試 interrupt（安全訊息打斷低優先度）")
    tts.speak("正在遞送零件，請稍候", priority=TTS_PRIORITY_TASK)
    time.sleep(0.1)
    tts.speak("緊急停止！", priority=TTS_PRIORITY_SAFETY, interrupt=True)
    time.sleep(3)

    logger.info("\n測試完成")


if __name__ == '__main__':
    test_tts()
