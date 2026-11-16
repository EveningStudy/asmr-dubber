"""Running a queue of smart plans."""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable, Sequence
from typing import Any

from ..errors import OperationCancelledError
from . import constants, failed_tasks, runlog, stages
from .constants import ReferenceEventCallback
from .domain import (
    AppConfig,
    SmartTaskPlan,
    ToolPaths,
)
from .errors import VideoPreparerError
from .plan_run import execute_prepared_smart_plan
from .runlog import append_log


def execute_smart_queue(
    paths: ToolPaths,
    config: AppConfig,
    plans: Sequence[SmartTaskPlan],
    *,
    executor: Callable[[ToolPaths, AppConfig, SmartTaskPlan], list[dict[str, Any]] | None]
    | None = None,
    reference_event_callback: ReferenceEventCallback | None = None,
) -> int:
    """Run configured works in order, retaining failures and continuing the queue."""

    print(f"\n全部 {len(plans)} 个作品已经配置完成，现在按顺序开始处理。")
    failures: list[tuple[SmartTaskPlan, str]] = []
    partials: list[tuple[SmartTaskPlan, list[dict[str, Any]]]] = []
    for index, plan in enumerate(plans, start=1):
        work_log = stages.work_log_path(constants.STATE_ROOT, plan.plan_id, plan.folder.name)
        runlog._WORK_LOG = work_log
        print("\n" + "=" * 68)
        print(f"  队列 {index}/{len(plans)} · {plan.folder.name}")
        print("=" * 68)
        outcome, failure_message = "completed", ""
        warnings: list[dict[str, Any]] = []
        if reference_event_callback is not None:
            reference_event_callback(
                {
                    "kind": "plan_started",
                    "plan_id": plan.plan_id,
                    "work": plan.folder.name,
                    "output": str(plan.output_root),
                }
            )
        try:
            if executor is None:
                warnings = execute_prepared_smart_plan(
                    paths,
                    config,
                    plan,
                    reference_event_callback=reference_event_callback,
                )
            else:
                warnings = executor(paths, config, plan) or []
            if warnings:
                outcome = "partial"
                partials.append((plan, warnings))
                runlog.log_event(
                    f"作品完成但有句子被跳过 source={plan.folder}：{stages.summarize(warnings)}"
                )
            failed_tasks.remove_failed_task(plan.plan_id)
            failed_tasks.remove_failed_task(plan.retry_of)
        except (KeyboardInterrupt, OperationCancelledError):
            outcome = "cancelled"
            raise
        except VideoPreparerError as exc:
            message = str(exc)
            outcome, failure_message = "failed", message
            failures.append((plan, message))
            failed_tasks.save_failed_task(plan, error=message)
            if plan.retry_of and plan.retry_of != plan.plan_id:
                failed_tasks.remove_failed_task(plan.retry_of)
            runlog.log_event(f"队列任务失败 source={plan.folder} error={message}；继续下一项")
            print(f"\n这个作品处理失败：{message}", file=sys.stderr)
            print("已有状态和成品已保留，继续处理下一个作品。", file=sys.stderr)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            outcome, failure_message = "failed", message
            failures.append((plan, message))
            failed_tasks.save_failed_task(plan, error=message)
            if plan.retry_of and plan.retry_of != plan.plan_id:
                failed_tasks.remove_failed_task(plan.retry_of)
            runlog.log_event(f"队列任务未预期失败 source={plan.folder} error={message}；继续下一项")
            append_log(traceback.format_exc())
            print(f"\n这个作品发生未预期错误：{message}", file=sys.stderr)
            print("已有状态和成品已保留，继续处理下一个作品。", file=sys.stderr)

        finally:
            if reference_event_callback is not None:
                reference_event_callback(
                    {
                        "kind": "plan_finished",
                        "plan_id": plan.plan_id,
                        "status": outcome,
                        "error": failure_message,
                        "warnings": warnings,
                        "log": str(work_log),
                        "work": plan.folder.name,
                        "output": str(plan.output_root),
                    }
                )
            runlog._WORK_LOG = None

    succeeded = len(plans) - len(failures) - len(partials)
    print("\n" + "=" * 68)
    print(
        f"队列结束：成功 {succeeded} 个，部分完成 {len(partials)} 个，失败 {len(failures)} 个。"
        if partials
        else f"队列结束：成功 {succeeded} 个，失败 {len(failures)} 个。"
    )
    for plan, records in partials:
        print(f"  - {plan.folder}（部分完成）：{stages.summarize(records)}")
    if failures:
        print("失败项目：", file=sys.stderr)
        for plan, message in failures:
            print(f"  - {plan.folder}: {message}", file=sys.stderr)
        print(f"详细日志：{constants.LOG_FILE}", file=sys.stderr)
        return 1
    return 0
