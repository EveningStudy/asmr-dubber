"""Settings surface: preview first, explicit project selection, no startup scan."""

from pathlib import Path

from .cache_cleanup import clean_caches, scan_caches
from .pipeline import default_projects_dir


def build_controls(gr, stored, busy):
    with gr.Tab("存储与清理", id="cache-cleanup"):
        gr.Markdown(
            "仅清理项目缓存；不删除项目、原素材、校对文字、逐句配音、最终输出、模型或运行环境。"
        )
        root = gr.Textbox(
            label="扫描项目目录", value=stored.projects_root or str(default_projects_dir())
        )
        categories = gr.CheckboxGroup(
            label="清理范围",
            value=["safe"],
            choices=[
                ("安全缓存：已完成分离的输入副本与分块", "safe"),
                ("可重建缓存：分析音频、RTF 参考与中文中间轨、混音背景", "rebuild"),
                ("人声分离结果：以后重新处理需再次运行分离模型", "separation"),
            ],
        )
        gr.Markdown(
            "未完成分离的断点默认保留。清理不可撤销，但所列缓存可重新计算；不会自动删除整个项目。"
        )
        scan = gr.Button("扫描缓存（不删除）")
        projects = gr.CheckboxGroup(label="选择要清理的项目", choices=[], value=[])
        preview = gr.Textbox(label="缓存扫描结果", lines=9, interactive=False)
        confirm = gr.Checkbox(label="我已确认清理范围；需要时允许重新计算缓存", value=False)
        clean = gr.Button("清理选中项目的缓存", variant="stop")
        plan = gr.State(None)

        def scan_action(path, kinds):
            if busy():
                raise gr.Error("请等待当前项目或批量任务结束后再扫描清理。")
            try:
                result = scan_caches(path, kinds or [])
            except Exception as exc:
                raise gr.Error(str(exc)) from exc
            choices = []
            lines = [f"目录：{result['root']}"]
            for entry in result["projects"]:
                size = sum(s[0] for s in entry["files"].values())
                label = f"{entry['path']} — {size / 1024**3:.2f} GiB"
                choices.append((label, entry["path"]))
                lines.append(f"{label}，{len(entry['files'])} 个文件")
                lines.extend(f"  {name}" for name in entry["files"])
            lines.extend(result["skipped"])
            return result, gr.update(choices=choices, value=[]), "\n".join(lines), False

        def clean_action(result, selected, agreed, path, kinds):
            if busy():
                raise gr.Error("请等待当前项目或批量任务结束后再扫描清理。")
            if (
                not result
                or result["root"] != str(Path(path).expanduser().absolute())
                or result["categories"] != (kinds or [])
            ):
                raise gr.Error("清理范围已变化，请重新扫描。")
            try:
                removed, skipped = clean_caches(result, selected or [], agreed)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc
            message = (
                f"已清理 {removed / 1024**3:.2f} GiB。项目与已有成品保留。请重新扫描后继续清理。"
            )
            if skipped:
                message += "\n跳过：\n" + "\n".join(skipped)
            return None, gr.update(choices=[], value=[]), message, False

        outputs = [plan, projects, preview, confirm]
        scan.click(
            scan_action,
            [root, categories],
            outputs,
            concurrency_id="runtime_mutation",
            concurrency_limit=1,
        )
        clean.click(
            clean_action,
            [plan, projects, confirm, root, categories],
            outputs,
            concurrency_id="runtime_mutation",
            concurrency_limit=1,
        )
