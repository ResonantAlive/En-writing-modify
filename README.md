# writing_modify — 作文批改应用

当前进度：M1–M5 全部完成（账户、上传分组、批改核心、本地识别 + 答题纸模板、教师复核与结果汇总）。

## 一键启动（推荐）

```bash
D:\Transformer\.venv\Scripts\python.exe D:\Transformer\writing_modify\start.py
```

或在资源管理器里双击 `start.bat`。脚本会：
1. 用项目虚拟环境 `D:\Transformer\.venv` 重新启动自己（用别的 Python 双击也行）；
2. 检查依赖和模型目录，**只报告缺什么、给出安装命令，从不自动安装**；
3. 启动服务（单 worker，默认只监听 `127.0.0.1:8000`）；
4. 后台预热本地识别引擎（第一次会把 PP-OCRv6 导出成 OpenVINO 模型到 `data/ov_models/`，约 30–45 秒，之后不到 1 秒）；
5. 服务就绪后自动打开浏览器，并在控制台显示邀请码文件的位置（不显示邀请码本身）。

参数：`--no-browser` 不开浏览器；`--no-warmup` 不预热；`--check` 只做检查不启动。

## 手动启动

```bash
cd D:\Transformer\writing_modify
D:\Transformer\.venv\Scripts\python.exe -m app
```

- 默认只监听 `127.0.0.1:8000`，**单 worker**。接口文档：`http://127.0.0.1:8000/docs`。
- 配置：`data/app/settings.json`（首次启动自动生成，缺省项取默认值）。
- 已有另一个服务进程在运行时，第二个进程会因拿不到执行器锁而拒绝启动。
- 跑测试：`D:\Transformer\.venv\Scripts\python.exe -m pytest`；前端纯逻辑测试：`node frontend/tests/m5_model.test.mjs`（另有 m3_model、frontend_model）

## 本地识别（路线 1 / 视觉路线的“先本地转写”）

- 模型：PP-OCRv6 medium（`PP-OCRv6_medium_det_safetensors`、`PP-OCRv6_medium_rec_safetensors`，只读）。只有这一个本地引擎。
- 运行时依赖 `openvino`；第一次导出模型时还需要 `torch` + `transformers`。导出的 `data/ov_models/` 不分平台，可以直接拷到别的机器（例如 Mac），那台机器就不用装 torch。
- 设置项 `ocr.backend`：`"openvino"`（默认）或 `"torch"`；`ocr.threads`：0 = 用全部核心。
- 质量规则：原分辨率检测、FP32，不缩图、不量化（作文里的拼写错误必须原样识别出来）。
- 评测：`D:\Transformer\.venv\Scripts\python.exe tools\ocr_eval.py`（默认读 `test_data/`，输出到 `eval_out/`）。
- 拍斜的照片会先按文字行角度自动转正再还原版式。

## 答题纸模板（本地识别必需）

IG 等答题纸上印着标题、说明、页边提示、条形码和数字、页码，本地识别会把它们一起读出来。所以使用本地识别前：

1. 打开“答题纸”页 → 新建模板（例如“IGCSE 0510 答题纸”）；
2. 上传同款**空白**答题纸的照片（一套有多页时，每种页面各拍一张；之后可以随时继续补充）；
3. 等每页显示“可用”（后台识别，一页约 10 秒），可展开查看识别到的印刷文字；
4. 上传作文时选择该模板。识别作文时会自动对齐并去掉这些印刷内容；批次详情里可以展开“已去除的答题纸印刷内容”核对。

模板可反复使用；不能删除，拍错的页可以“停用”。root 可以把模板共享给所有人。
核对真实照片：`D:\Transformer\.venv\Scripts\python.exe tools\sheet_check.py --blank 空白页.jpg 作文照片.jpg`

## 教师复核与结果汇总

在批次详情的作文列表里点“查看结果 / 复核”：

- **一键采纳 AI 结果**：最终分 = AI 分；
- **教师评分**：填分数（0 到满分），可选添加意见条目（名称可空；最多 200 条，每条最多 10000 字，emoji 算 1 个字）；
- AI 批改失败的作文也可以直接人工评分（没有 AI 分，差值为空）；
- 每次保存都新增一版，旧版本和 AI 原始结果都不会被改动，展开“复核历史”可以查看；
- 复核结果会作为以后同类作文的范例（few-shot），只用于本人的批次。

批次完成或取消后，详情页顶部显示“批次汇总”：已复核、待复核、失败、已取消的篇数，平均最终分，复核与 AI 的平均差。
对应文件 `data/history/…/<task_code>/result-<task_code>.json` 在每次复核后自动更新，覆盖前把旧文件备份到同目录的 `result_backup/`（保留最近 20 份）；也可以手动“重新生成”或“下载汇总 JSON”。

## 邀请码（重要）

首次启动会生成 `data/app/invite_codes.json`（含 `root_invite_code` 与 `user_invite_code`）。
控制台和日志只会提示文件路径，**任何 API 都不会返回邀请码**。root 重置邀请码后，接口只返回“新码已写入服务器上的该文件”，
**请到服务器上打开这个文件查看新码**。重启不会覆盖已有文件。

## 目录

```
app/        后端代码
data/       运行数据（已 gitignore，含明文 key、邀请码、学生数据；不得对外暴露）
  app/      settings / users / invite_codes / providers / prompts / sheet_templates(.json) ...
  state/    counters.json、auth_state.json、locks/
  staging/  上传暂存   history/  历史批改   logs/{app,audit}
tests/      测试（与 data/ 隔离，不含真实学生数据）
DECISIONS.md  实现细节决定
```

## API 示例（curl）

```bash
# 注册（邀请码从 data/app/invite_codes.json 取）
curl -X POST http://127.0.0.1:8000/api/v1/auth/register -H "Content-Type: application/json" -d "{\"invite_code\":\"XXXX-...\",\"username\":\"teacher1\",\"password\":\"a-long-password\"}"
# 登录并保存 Cookie
curl -c jar.txt -X POST http://127.0.0.1:8000/api/v1/auth/login -H "Content-Type: application/json" -d "{\"username\":\"teacher1\",\"password\":\"a-long-password\"}"
# 创建 API 令牌（Cookie 写请求需带 X-CSRF-Token，值取自登录响应的 csrf_token）
curl -b jar.txt -X POST http://127.0.0.1:8000/api/v1/tokens -H "X-CSRF-Token: <csrf_token>" -H "Content-Type: application/json" -d "{\"name\":\"script\",\"role\":\"user\"}"
# 用令牌访问（Bearer 不需要 CSRF）
curl http://127.0.0.1:8000/api/v1/auth/me -H "Authorization: Bearer wm_..."
# 教师复核：一键采纳 / 教师评分（可带条目）
curl -X POST http://127.0.0.1:8000/api/v1/batches/<task_code>/subtasks/<subtask_id>/review -H "Authorization: Bearer wm_..." -H "Content-Type: application/json" -d "{\"adopt_ai\":true}"
curl -X POST http://127.0.0.1:8000/api/v1/batches/<task_code>/subtasks/<subtask_id>/review -H "Authorization: Bearer wm_..." -H "Content-Type: application/json" -d "{\"final_score\":15.5,\"items\":[{\"name\":\"语言\",\"content\":\"时态错误较多\"}]}"
# 查看复核历史 / 批次汇总 / 手动重新生成汇总
curl http://127.0.0.1:8000/api/v1/batches/<task_code>/subtasks/<subtask_id>/review -H "Authorization: Bearer wm_..."
curl http://127.0.0.1:8000/api/v1/batches/<task_code>/result -H "Authorization: Bearer wm_..."
curl -X POST http://127.0.0.1:8000/api/v1/batches/<task_code>/result/regenerate -H "Authorization: Bearer wm_..."
```

## 手动删除数据

第一版不提供删除功能。需要删除时：**先停止服务**，再删除对应目录（如 `data/history/YYYY/MM/DDMMYYYY/<task_code>/`）。不要在服务运行时删。

## 局域网部署

- 监听非回环地址必须启用 HTTPS（`network.https=true`，自带证书 `ssl_certfile/ssl_keyfile`，或放在 HTTPS 反向代理后面），否则拒绝启动；
  仅在显式设置 `network.allow_insecure_lan=true` 时放行，并在控制台警告（不推荐）。
- 非回环监听时会话 Cookie 自动加 `Secure`。
- 经反向代理时，把代理地址写进 `network.trusted_proxies`，只有来自这些地址的 `X-Forwarded-For` 才会被采信（用于登录锁定与限流）。
- 前后端分离部署时，把前端源写进 `network.cors_origins`。
