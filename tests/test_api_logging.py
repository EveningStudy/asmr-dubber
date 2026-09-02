import logging

import httpx

from asmr_dubber.api_logging import logged_http_client, safe_api_url


def test_safe_api_url_removes_credentials_query_and_fragment() -> None:
    assert (
        safe_api_url("https://user:secret@example.com:8443/v1/test?token=secret#fragment")
        == "https://example.com:8443/v1/test"
    )
    assert safe_api_url("https://example.com:broken/path") == "无效地址"
    assert safe_api_url("secret-token") == "无效地址"


def test_logged_http_client_records_metadata_without_secrets(caplog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-request-id": "request-123"},
            request=request,
            json={"ok": True},
        )

    caplog.set_level(logging.INFO, logger="asmr_dubber.api")
    with logged_http_client(
        "测试服务",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer top-secret"},
    ) as client:
        response = client.get("https://user:password@example.test/v1?q=secret")

    assert response.status_code == 200
    text = caplog.text
    assert "测试服务" in text
    assert "状态=200" in text
    assert "请求ID=request-123" in text
    assert "https://example.test/v1" in text
    for secret in ("top-secret", "password", "q=secret"):
        assert secret not in text
