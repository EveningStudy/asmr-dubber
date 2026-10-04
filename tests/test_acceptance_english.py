import base64
import json
import shutil
import subprocess
from importlib.resources import files

import pytest

from asmr_dubber.localization import catalog


def test_english_localizes_saved_keys_model_details_and_sentence_controls():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute the native browser localization module")
    script = files("asmr_dubber").joinpath("frontend/session.js").read_bytes()
    uri = "data:text/javascript;base64," + base64.b64encode(script).decode()
    examples = [
        "原声开关（仅分离时）",
        "原声音量微调（dB）",
        "中文音量微调（dB）",
        "已将 DeepSeek（推荐默认） 的 API Key 以便携式明文保存在程序目录中；"
        "删除程序文件夹时会一并删除。",
        "当前服务尚未保存 API Key。",
        "IndexTTS2 已就绪：D:/AT01/indextts2.exe；模型目录：D:/AT01/checkpoints",
        "字幕生成完成",
        "failed",
        "放置中文句子 s000044（自动加速 1.23×）",
        "项目语言：日语\n总句数：54",
    ]
    code = (
        f"const m=await import({json.dumps(uri)});"
        "m.state.language='en';"
        f"m.state.boot={{locales:{json.dumps(catalog(), ensure_ascii=False)},"
        f"labels:{json.dumps(catalog('zh'), ensure_ascii=False)}}};"
        f"console.log(JSON.stringify({json.dumps(examples, ensure_ascii=False)}.map(m.t)));"
    )
    result = subprocess.run(
        [node, "--input-type=module"],
        input=code,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    values = json.loads(result.stdout)
    assert all(not any("\u4e00" <= c <= "\u9fff" for c in value) for value in values), values
    assert "D:/AT01/indextts2.exe" in values[5]
    assert "D:/AT01/checkpoints" in values[5]
