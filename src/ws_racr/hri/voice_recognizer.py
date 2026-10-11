"""
語音轉文字模組 - 使用 sounddevice
支援按鍵觸發的語音識別功能
"""
import logging
import sounddevice as sd
import speech_recognition as sr
import scipy.io.wavfile as wavfile
import numpy as np
import threading
import time
import io
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class VoiceRecognizerSD:
    """語音識別器 - 使用 sounddevice 進行錄音"""
    
    def __init__(self, language='zh-TW', device_index=None):
        """
        初始化語音識別器
        
        Args:
            language: 語言設定 ('zh-TW', 'en-US', 'zh-CN')
            device_index: 音頻設備索引，None 表示使用預設設備
        """
        self.recognizer = sr.Recognizer()
        self.language = language
        self.device_index = device_index
        
        # 錄音控制
        self.is_recording = False
        self.recording_thread = None
        self.recorded_audio = None
        
        # 回調函數
        self.on_result_callback = None
        self.on_error_callback = None
        self.on_start_recording = None
        self.on_stop_recording = None
        
        # 獲取設備資訊
        try:
            if device_index is not None:
                # 使用指定的設備
                device_info = sd.query_devices(device_index, 'input')
                self.device_index = device_index
            else:
                # 自動選擇最佳設備
                # 先嘗試找支援 16000 Hz 的設備（語音識別標準）
                logger.info("自動偵測最佳麥克風...")
                best_device = None
                
                for i in range(len(sd.query_devices())):
                    device = sd.query_devices(i)
                    if device['max_input_channels'] > 0:
                        # 優先選擇支援 16000 Hz 的設備
                        if abs(device['default_samplerate'] - 16000) < 100:
                            best_device = i
                            logger.info(f"   找到 16000 Hz 設備: [{i}] {device['name']}")
                            break
                
                if best_device is not None:
                    self.device_index = best_device
                    device_info = sd.query_devices(best_device, 'input')
                    logger.info(f"   使用最佳設備: {device_info['name']}")
                else:
                    # 沒找到，使用預設設備
                    device_info = sd.query_devices(kind='input')
                    self.device_index = device_info.get('index')
                    logger.info(f"   ℹ使用預設設備: {device_info['name']}")
            
            self.sample_rate = int(device_info['default_samplerate'])
            self.channels = min(1, device_info['max_input_channels'])  # 使用單聲道
            
            logger.info(f"音頻設備 [{self.device_index}]: {device_info['name']}")
            logger.info(f"採樣率: {self.sample_rate} Hz")
            logger.info(f"語音識別器已初始化 (sounddevice)")
            logger.info(f"語言設定: {language}")
            
        except Exception as e:
            logger.warning(f"設備初始化警告: {e}")
            # 使用預設值
            self.sample_rate = 48000
            self.channels = 1
            logger.info(f"使用預設設定: {self.sample_rate} Hz, {self.channels} 通道")
    
    # Domain-specific phonetic corrections for zh-TW robot commands.
    # Google STT frequently mishears "夾" (clamp/grasp) as "甲" or "加".
    _PHONETIC_CORRECTIONS: tuple = (
        ("甲取", "夾取"),
        ("甲曲", "夾取"),
        ("加取", "夾取"),
        ("甲擊", "夾取"),
        ("甲起", "夾取"),
        ("甲緊", "夾緊"),
        ("加緊", "夾緊"),
        ("甲爪", "夾爪"),   # 放開甲爪 → 放開夾爪
    )

    def _align_to_ontology(self, text: str) -> str:
        """Apply domain phonetic corrections to raw ASR output.

        Replaces known misrecognitions caused by phonetic similarity
        in zh-TW speech (e.g. '甲取' → '夾取').
        """
        for wrong, correct in self._PHONETIC_CORRECTIONS:
            if wrong in text:
                text = text.replace(wrong, correct)
                logger.info(f"STT 對齊: '{wrong}' → '{correct}'")
        return text

    def set_callbacks(self, 
                     on_result: Optional[Callable[[str], None]] = None,
                     on_error: Optional[Callable[[str], None]] = None,
                     on_start_recording: Optional[Callable[[], None]] = None,
                     on_stop_recording: Optional[Callable[[], None]] = None):
        """設定回調函數"""
        self.on_result_callback = on_result
        self.on_error_callback = on_error
        self.on_start_recording = on_start_recording
        self.on_stop_recording = on_stop_recording
    
    def start_recording(self):
        """開始錄音"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.recorded_audio = None
        
        # 啟動錄音線程
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
        
        if self.on_start_recording:
            self.on_start_recording()
        
        logger.info("開始錄音...")
    
    def stop_recording(self):
        """停止錄音並進行語音識別"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        if self.on_stop_recording:
            self.on_stop_recording()
        
        logger.info("停止錄音，開始識別...")
        
        # 等待錄音線程結束
        if self.recording_thread:
            self.recording_thread.join(timeout=2)
        
        # 處理音頻
        if self.recorded_audio is not None:
            threading.Thread(target=self._process_audio, daemon=True).start()
        else:
            if self.on_error_callback:
                self.on_error_callback("沒有錄到音頻")
            else:
                logger.warning("沒有錄到音頻")
    
    def _record_audio(self):
        """錄音線程 - 持續錄音直到停止"""
        try:
            audio_chunks = []

            def audio_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"錄音狀態: {status}")
                audio_chunks.append(indata.copy())  # 永遠 append；by InputStream 關閉控制終止

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16',
                callback=audio_callback,
                device=self.device_index
            ):
                while self.is_recording:
                    time.sleep(0.05)
                time.sleep(0.15)  # 確保最後幾個 frame 被 callback 收到再關閉 stream

            if audio_chunks:
                self.recorded_audio = np.concatenate(audio_chunks, axis=0)
                duration = len(self.recorded_audio) / self.sample_rate
                logger.info(f"錄音長度: {duration:.2f} 秒")
                if duration < 0.8:
                    logger.warning(f"錄音太短 ({duration:.2f}s)，可能未說話")
            else:
                logger.warning("沒有錄到音頻數據")

        except Exception as e:
            error_msg = f"錄音錯誤: {e}"
            if self.on_error_callback:
                self.on_error_callback(error_msg)
            else:
                logger.error(f"{error_msg}")
    
    def _process_audio(self):
        """處理音頻並進行語音識別"""
        try:
            if self.recorded_audio is None or len(self.recorded_audio) == 0:
                error_msg = "沒有音頻數據"
                if self.on_error_callback:
                    self.on_error_callback(error_msg)
                else:
                    logger.warning(f"{error_msg}")
                return
            
            logger.info("正在識別語音...")
            
            # 將 numpy array 轉換為 WAV 格式
            wav_io = io.BytesIO()
            wavfile.write(wav_io, self.sample_rate, self.recorded_audio)
            wav_io.seek(0)
            
            # 使用 speech_recognition 識別
            with sr.AudioFile(wav_io) as source:
                audio = self.recognizer.record(source)
            
            # Google 語音識別（使用線程+超時機制強制中斷）
            max_retries = 2
            timeout = 15.0  # 15 秒超時（Google API 通常 5-10 秒內完成）
            
            def recognize_with_timeout():
                """在單獨線程中執行識別"""
                try:
                    return self.recognizer.recognize_google(
                        audio, 
                        language=self.language,
                        show_all=False
                    )
                except sr.UnknownValueError:
                    return "UNKNOWN_VALUE_ERROR"
                except sr.RequestError as e:
                    return f"REQUEST_ERROR: {e}"
                except Exception as e:
                    return f"EXCEPTION: {e}"
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        logger.info(f"   重試 {attempt}/{max_retries}...")
                    
                    # 使用線程執行，設置超時
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(recognize_with_timeout)
                        
                        try:
                            # 等待結果，最多 timeout 秒
                            result = future.result(timeout=timeout)
                            
                            # 檢查結果類型
                            if isinstance(result, str):
                                if result == "UNKNOWN_VALUE_ERROR":
                                    error_msg = "無法理解語音內容（請說清楚一點）"
                                    if self.on_error_callback:
                                        self.on_error_callback(error_msg)
                                    else:
                                        logger.warning(f"{error_msg}")
                                    return None
                                elif result.startswith("REQUEST_ERROR:"):
                                    error_msg = result.replace("REQUEST_ERROR: ", "")
                                    if attempt < max_retries - 1:
                                        logger.warning(f"語音識別服務錯誤: {error_msg}，準備重試...")
                                        time.sleep(1)
                                        continue
                                    else:
                                        if self.on_error_callback:
                                            self.on_error_callback(f"語音識別服務錯誤: {error_msg}")
                                        else:
                                            logger.error(f"語音識別服務錯誤: {error_msg}")
                                        return None
                                elif result.startswith("EXCEPTION:"):
                                    error_msg = result.replace("EXCEPTION: ", "")
                                    if self.on_error_callback:
                                        self.on_error_callback(f"識別過程異常: {error_msg}")
                                    else:
                                        logger.error(f"識別過程異常: {error_msg}")
                                    return None
                                else:
                                    # 正常的識別結果 — apply phonetic corrections
                                    result = self._align_to_ontology(result)
                                    logger.info(f"識別結果: '{result}'")
                                    if self.on_result_callback:
                                        self.on_result_callback(result)
                                    return result
                            else:
                                error_msg = "識別結果為空"
                                if self.on_error_callback:
                                    self.on_error_callback(error_msg)
                                else:
                                    logger.warning(f"{error_msg}")
                                return None
                        
                        except concurrent.futures.TimeoutError:
                            # 超時：強制取消任務
                            logger.warning(f"語音識別超時 ({timeout}秒)")
                            future.cancel()
                            
                            if attempt < max_retries - 1:
                                logger.info(f"   準備重試...")
                                continue
                            else:
                                error_msg = f"語音識別超時（{timeout}秒後無響應）"
                                if self.on_error_callback:
                                    self.on_error_callback(error_msg)
                                else:
                                    logger.error(f"{error_msg}")
                                return None
                
                except Exception as e:
                    error_msg = f"識別過程異常: {e}"
                    if self.on_error_callback:
                        self.on_error_callback(error_msg)
                    else:
                        logger.error(f"{error_msg}")
                    return None
            
            # 所有重試都失敗
            error_msg = "語音識別失敗（已重試所有次數）"
            if self.on_error_callback:
                self.on_error_callback(error_msg)
            else:
                logger.error(f"{error_msg}")
            return None
                    
        except Exception as e:
            error_msg = f"語音處理錯誤: {e}"
            if self.on_error_callback:
                self.on_error_callback(error_msg)
            else:
                logger.error(f"{error_msg}")
        finally:
            # 清理錄音數據，準備下次使用
            self.recorded_audio = None
            logger.info("語音識別器就緒，可再次使用")
    
    def record_and_recognize(self, duration: float = 5.0) -> Optional[str]:
        """
        同步語音識別 - 錄製指定時間後自動識別
        
        Args:
            duration: 錄音時長（秒）
            
        Returns:
            識別到的文字，失敗返回 None
        """
        try:
            logger.info(f"開始錄音 {duration} 秒...")
            
            # 使用 sounddevice 錄音
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16',
                device=self.device_index
            )
            sd.wait()  # 等待錄音完成
            
            logger.info("正在識別語音...")
            
            # 轉換為 WAV 格式
            wav_io = io.BytesIO()
            wavfile.write(wav_io, self.sample_rate, recording)
            wav_io.seek(0)
            
            # 使用 speech_recognition 識別
            with sr.AudioFile(wav_io) as source:
                audio = self.recognizer.record(source)
            
            # 添加超時避免卡住
            text = self.recognizer.recognize_google(
                audio, 
                language=self.language,
                show_all=False
            )
            
            if text:
                text = self._align_to_ontology(text)
                logger.info(f"識別結果: '{text}'")
                return text
            else:
                logger.warning("識別結果為空")
                return None
                
        except sr.UnknownValueError:
            logger.warning("無法理解語音內容")
            return None
        except sr.RequestError as e:
            logger.error(f"語音識別服務錯誤: {e}")
            return None
        except Exception as e:
            logger.error(f"語音識別錯誤: {e}")
            return None
    
    def test_microphone(self):
        """測試麥克風是否正常工作"""
        logger.info("麥克風測試...")
        logger.info("請說話，將錄製 3 秒鐘...")
        
        result = self.record_and_recognize(duration=3.0)
        
        if result:
            logger.info(f"麥克風測試成功！識別到: '{result}'")
            return True
        else:
            logger.error("麥克風測試失敗")
            return False
    
    def cleanup(self):
        """清理資源"""
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1)
        logger.info("語音識別器已清理")


def main():
    """測試語音識別功能"""
    logger.info("語音識別測試程式 (sounddevice)")
    logger.info("按 Enter 開始錄音，再按 Enter 停止錄音")
    logger.info("輸入 'q' 退出")
    logger.info("-" * 50)
    
    # 初始化語音識別器
    voice = VoiceRecognizerSD(language='zh-TW')
    
    # 設定回調函數
    def on_result(text):
        logger.info(f"您說了: '{text}'")
    
    def on_error(error):
        logger.error(f"錯誤: {error}")
    
    def on_start():
        logger.info("錄音中...")
    
    def on_stop():
        logger.info("錄音結束，處理中...")
    
    voice.set_callbacks(
        on_result=on_result,
        on_error=on_error,
        on_start_recording=on_start,
        on_stop_recording=on_stop
    )
    
    # 測試麥克風
    voice.test_microphone()
    
    logger.info("\n準備就緒！按 Enter 開始/停止錄音...")
    
    try:
        while True:
            user_input = input().strip()
            
            if user_input.lower() == 'q':
                logger.info("\n退出語音識別")
                break
            
            # 切換錄音狀態
            if voice.is_recording:
                voice.stop_recording()
            else:
                voice.start_recording()
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("\n程式中斷")
    finally:
        voice.cleanup()


if __name__ == "__main__":
    main()
