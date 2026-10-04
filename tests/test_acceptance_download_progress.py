from asmr_dubber.services import models
from asmr_dubber.services.application import Application
from tests.test_native_services import completed


def test_download_logs_supply_numeric_progress_and_keep_it_between_messages(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path))
    events = []

    def install(_backend, **kwargs):
        kwargs["log_callback"]("下载进度：35.6%")
        kwargs["log_callback"]("正在校验下载文件")
        return "已安装"

    monkeypatch.setattr(models.runtime, "install_backend", install)
    models.download({"model": "indextts2"}, lambda *args, **kw: events.append(args), None)
    assert events[0][1:] == (35.6, 100)
    assert events[1][1:] == (35.6, 100)


def test_successful_task_finishes_its_progress_without_an_explicit_final_callback(tmp_path):
    from asmr_dubber.services.tasks import Tasks

    tasks = Tasks(lambda *_: {}, tmp_path / "tasks.json")
    task = completed(tasks, tasks.start({"kind": "import"}, "project")["id"])
    assert task["status"] == "completed"
    assert task["current"] == task["total"] == 1


def test_curl_transfer_progress_is_visible_in_api_task(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path))

    def install(_backend, **kwargs):
        kwargs["log_callback"](" 35  2.70G  35 984.9M   0 0 42.22M 0 00:01")
        kwargs["log_callback"]("正在导入 Windows 进阶依赖包...")
        raise RuntimeError("stop after progress")

    monkeypatch.setattr(models.runtime, "install_backend", install)
    app = Application(tmp_path / "tasks.json")
    task = completed(app.tasks, app.start({"kind": "download", "model": "faster_whisper"})["id"])
    assert task["status"] == "failed"
    assert (task["current"], task["total"]) == (35, 100)
