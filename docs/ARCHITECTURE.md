中文 | [English](en/ARCHITECTURE.md)

[文档索引](INDEX.md) · [README](../README.md)

# 原生界面与服务架构

用户操作见[图文手册](USER_GUIDE.md)。当前界面为原生 HTML/CSS/JS，没有前端构建工具或 Gradio 依赖。

## 调用路径

```text
Windows EXE / scripts
  → ui.py → http_server.py → api.py / api_contract.py
  → services/（项目、设置、模型、批量、任务）
  → pipeline / asr / translation / tts / audio / autoflow
  → 项目与便携数据

frontend/*.js → JSON API → services → 核心
CLI → 核心项目流程
```

接口层校验请求与绑定服务，不实现业务流程；服务层不依赖界面库，组合核心模块和持久状态。浏览器用会话 token、媒体能力 URL 与 JSON 数据，服务器保持 revision/任务锁保护。

## 模块职责

| 位置 | 职责 |
|---|---|
| frontend/index.html/styles.css | 页面外壳、布局 |
| frontend/projects.js/project_dialogs.js | 四步工作区、字幕/参考/复核弹窗 |
| frontend/forms.js | 参数清单驱动表单与自动保存 |
| frontend/session.js/app.js | 会话、JSON 请求、保存队列、任务轮询、导航、本地化 |
| frontend/models.js/batch.js/settings.js | 模型、队列、设置/诊断/清理页面 |
| api_contract.py/api_contracts.py/api.py | 请求类型、路由与参数校验 |
| services/parameters.py/parameter_layout.json | 参数类型/默认值/范围/分组/显示条件；与领域模型合成一份清单 |
| services/settings.py | 全局/项目范围、Key 与持久设置 |
| services/projects.py/project_* | 项目视图、编辑、阶段操作、音频参考 |
| services/review.py | 复核报告、采纳/锁定/撤销 |
| services/models.py/model_status.py | 硬件、权重/依赖状态、下载/移除/离线包 |
| services/batch*.py | 扫描、计划快照、队列、执行 |
| services/tasks.py | 启动、进度、取消、重启中断与恢复；保留 50 个已结束任务 |
| lifecycle.py/models.py/storage.py | 缓存失效、项目验证、revision、跨进程锁与原子写入 |
| model_registry.py/runtime_manager.py | 后端能力、安装/检测合同 |

## 持久化与资源

项目拥有自己的设置；默认值只在创建时复制。密钥单独存全局，不放任务快照；项目任务结果只记录 manifest，页面重新载入项目。已消费上传清理，废弃上传过期回收。媒体注册使用规范路径映射。

安装、推理与批量受互斥保护。取消通过 token/子进程控制向下传播，核心检查点保留有效完成结果；重启将活跃任务标记 interrupted。缓存签名包括相关文本、时间、参考、模型与参数，避免误复用。

清理方案的高精度时间戳/inode 以字符串跨 JSON，清理时重新验证指纹和锁。模型按登记文件与完整性检查判断就绪/删除，不能用单个同名文件代替整个模型。

## HTTP 边界

默认 loopback Host 白名单；远程绑定强制 Basic 登录。写请求验证 `X-ASMR-Token` 和 Origin；媒体 URL 只访问注册资源。请求/上传有大小限制，模型/项目操作需通过服务验证；不是任意文件 Web 服务。

开发测试运行 pytest、Ruff、Pyright；真实浏览器回归另运行 `tests/native_viewport.cjs` 和 `tests/native_cleanup.cjs`。短格式/契约测试不能代替模型真实推理或听感评价。运行时 prompts 是代码资产，文档整理不改变它们，见[PROMPTS](PROMPTS.md)。
