import struct
from pathlib import Path

from asmr_dubber import dll_dependencies as deps
from asmr_dubber import runtime_health as health


def _fake_pe(path: Path, imports: list[str]) -> None:
    """Smallest PE32+ file whose import table names ``imports``."""

    header, optional_size, raw, rva = 0x40, 240, 0x200, 0x1000
    names = bytearray()
    offsets = []
    descriptors_size = 20 * (len(imports) + 1)
    for name in imports:
        offsets.append(descriptors_size + len(names))
        names += name.encode() + b"\0"
    section = bytearray()
    for offset in offsets:
        section += struct.pack("<IIIII", 0, 0, 0, rva + offset, 0)
    section += b"\0" * 20 + names
    data = bytearray(raw + len(section))
    data[0:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, header)
    data[header : header + 4] = b"PE\0\0"
    struct.pack_into("<H", data, header + 6, 1)
    struct.pack_into("<H", data, header + 20, optional_size)
    optional = header + 24
    struct.pack_into("<H", data, optional, 0x20B)
    struct.pack_into("<I", data, optional + 112 + 8, rva)
    table = optional + optional_size
    struct.pack_into("<II", data, table + 8, len(section), rva)
    struct.pack_into("<I", data, table + 20, raw)
    data[raw:] = section
    path.write_bytes(bytes(data))


def _windows(tmp_path: Path, monkeypatch, present: list[str]) -> None:
    system = tmp_path / "Windows" / "System32"
    system.mkdir(parents=True)
    for name in present:
        (system / name).write_bytes(b"")
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path / "Windows"))
    monkeypatch.setenv("PATH", "")


def test_missing_dll_is_named_with_the_file_that_needs_it(tmp_path, monkeypatch):
    _windows(tmp_path, monkeypatch, ["KERNEL32.dll"])
    bundle = tmp_path / "bin"
    bundle.mkdir()
    _fake_pe(
        bundle / "tool.exe",
        ["helper.dll", "VCRUNTIME140.dll", "KERNEL32.dll", "api-ms-win-crt-runtime-l1-1-0.dll"],
    )
    _fake_pe(bundle / "helper.dll", ["MFPlat.DLL", "python311.dll", "KERNEL32.dll"])

    missing = deps.missing_dlls([bundle / "tool.exe"], [bundle])

    assert missing == {"VCRUNTIME140.dll": ["tool.exe"], "MFPlat.DLL": ["helper.dll"]}
    report = "\n".join(deps.describe_missing(missing))
    assert "缺少 MFPlat.DLL（helper.dll 需要）" in report
    assert deps.ADVICE_MEDIA in report and deps.ADVICE_VC in report


def test_nothing_missing_is_said_plainly(tmp_path, monkeypatch):
    _windows(tmp_path, monkeypatch, ["KERNEL32.dll", "nvcuda.dll"])
    _fake_pe(tmp_path / "tool.exe", ["KERNEL32.dll", "nvcuda.dll"])
    assert deps.describe_missing(deps.missing_dlls([tmp_path / "tool.exe"])) == [
        "DLL 依赖检查：所需 DLL 都能找到。"
    ]


def test_unreadable_files_are_not_guessed_about(tmp_path):
    (tmp_path / "broken.dll").write_bytes(b"not a program")
    (tmp_path / "empty.dll").write_bytes(b"")
    assert deps.pe_imports(tmp_path / "broken.dll") == []
    assert deps.pe_imports(tmp_path / "empty.dll") == []


def test_advice_matches_the_kind_of_dll():
    assert deps.advice_for("vcomp140.dll") == deps.ADVICE_VC
    assert deps.advice_for("MFReadWrite.dll") == deps.ADVICE_MEDIA
    assert deps.advice_for("nvcuda.dll") == deps.ADVICE_NVIDIA
    assert deps.advice_for("cudnn_ops64_9.dll") == deps.ADVICE_CUDA
    assert deps.advice_for("ggml-base.dll") == deps.ADVICE_OTHER


def test_python_backend_failure_scans_the_dll_named_in_the_error(tmp_path, monkeypatch):
    _windows(tmp_path, monkeypatch, ["KERNEL32.dll"])
    lib = tmp_path / "site-packages" / "torch" / "lib"
    lib.mkdir(parents=True)
    _fake_pe(lib / "c10.dll", ["KERNEL32.dll", "VCOMP140.DLL"])
    message = (
        "未通过：退出码 1\nOSError: [WinError 126] 找不到指定的模块。 "
        f'Error loading "{lib / "c10.dll"}" or one of its dependencies.'
    )
    if lib.drive:  # the error text carries a Windows path only on Windows
        report = "\n".join(health._dependency_report("indextts2", message))
        assert "缺少 VCOMP140.DLL（c10.dll 需要）" in report
    assert health._dependency_report("indextts2", "通过：启动/依赖导入成功") == []


def test_parakeet_report_reads_the_installed_crispasr(tmp_path, monkeypatch):
    _windows(tmp_path, monkeypatch, ["KERNEL32.dll"])
    home = tmp_path / "home"
    bundle = home / "runtimes" / "crispasr" / "bin"
    bundle.mkdir(parents=True)
    _fake_pe(bundle / "crispasr.exe", ["crispasr.dll", "KERNEL32.dll"])
    _fake_pe(bundle / "crispasr.dll", ["MFReadWrite.dll"])
    monkeypatch.setattr(health, "portable_home", lambda: home)
    report = "\n".join(health._dependency_report("parakeet_nemo", "通过"))
    assert "缺少 MFReadWrite.dll（crispasr.dll 需要）" in report
    monkeypatch.setattr(health, "portable_home", lambda: tmp_path / "nowhere")
    assert health._dependency_report("parakeet_nemo", "未通过") == []
