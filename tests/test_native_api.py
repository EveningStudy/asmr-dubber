import threading

import httpx
import pytest

from asmr_dubber.http_server import Server
from asmr_dubber.services.application import Application


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    application = Application(tmp_path / "tasks.json")
    server = Server(("127.0.0.1", 0), application)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(3)


def test_http_json_validation_and_global_autosave(server):
    backend, url = server
    headers = {"X-ASMR-Token": backend.token}
    with httpx.Client(trust_env=False) as client:
        response = client.post(
            url + "/api/settings/update", json={"changes": {"tts_speed": 1.3}}, headers=headers
        )
        assert response.status_code == 200
        assert response.json()["tts_speed"] == 1.3
        assert (
            client.get(url + "/api/bootstrap", headers=headers).json()["settings"]["tts_speed"]
            == 1.3
        )
        response = client.post(
            url + "/api/settings/update", json={"changes": {"tts_speed": 20}}, headers=headers
        )
        assert response.status_code == 422
        assert client.get(url + "/api/settings/get", headers=headers).json()["tts_speed"] == 1.3
        assert (
            client.post(url + "/api/settings/update", json=[1], headers=headers).status_code == 400
        )
        assert (
            client.post(
                url + "/api/tasks/start", json={"kind": "synthesize"}, headers=headers
            ).status_code
            == 422
        )


def test_http_session_and_origin_protect_writes(server):
    backend, url = server
    with httpx.Client(trust_env=False) as client:
        assert (
            client.post(
                url + "/api/settings/update", json={"changes": {"tts_speed": 1.4}}
            ).status_code
            == 403
        )
        assert (
            client.post(
                url + "/api/settings/update",
                json={"changes": {}},
                headers={"X-ASMR-Token": backend.token, "Origin": "https://attacker.invalid"},
            ).status_code
            == 403
        )
        assert (
            client.get(
                url + "/api/bootstrap",
                headers={"X-ASMR-Token": backend.token, "Host": "attacker.invalid"},
            ).status_code
            == 403
        )


def test_http_registered_media_ranges_and_no_arbitrary_files(server, tmp_path):
    backend, url = server
    file = tmp_path / "audio.wav"
    file.write_bytes(bytes(range(100)))
    media = backend.application.media.register(file)
    with httpx.Client(trust_env=False) as client:
        result = client.get(url + media, headers={"Range": "bytes=20-29"})
        assert result.status_code == 206
        assert result.headers["Content-Range"] == "bytes 20-29/100"
        assert result.content == bytes(range(20, 30))
        assert client.get(url + media, headers={"Range": "bytes=-3"}).content == bytes(
            range(97, 100)
        )
        assert client.get(url + media, headers={"Range": "bytes=500-600"}).status_code == 416
        assert client.get(url + "/media/not-registered").status_code == 404
        assert (
            client.get(url + "/api/not-found", headers={"X-ASMR-Token": backend.token}).status_code
            == 404
        )


def test_http_upload_and_secrets_are_not_returned(server):
    backend, url = server
    with httpx.Client(trust_env=False) as client:
        headers = {"X-ASMR-Token": backend.token, "X-Filename": "audio.wav"}
        upload = client.post(url + "/api/uploads", content=b"sample", headers=headers)
        assert upload.status_code == 200
        assert client.get(url + upload.json()["url"]).content == b"sample"
        headers = {"X-ASMR-Token": backend.token}
        secret = "test-private-value"
        result = client.post(
            url + "/api/settings/key",
            json={"kind": "translation", "name": "deepseek", "value": secret},
            headers=headers,
        )
        assert result.status_code == 200
        assert secret not in result.text
        assert secret not in client.get(url + "/api/bootstrap", headers=headers).text
        assert not backend.application.tasks.list()
