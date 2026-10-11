"""Small text helpers for speech-to-text output."""

import re


# 語音常誤：「第N部」→「第N步」（僅在「需要材料／做什麼」等組裝問句語境）
_STT_STEP_BU_RE = re.compile(r"第([\d一二三四五六七八九十百零〇两兩]+)部")
_STT_ASSEMBLY_NEED_RE = re.compile(
    r"需要什麼|需要哪些|材料|零件|工具|怎麼做|要做什麼|準備什麼|要什麼",
)


def normalize_stt_assembly_query(text: str) -> str:
    if not text or "部" not in text:
        return text
    if not _STT_ASSEMBLY_NEED_RE.search(text):
        return text
    return _STT_STEP_BU_RE.sub(r"第\1步", text)
