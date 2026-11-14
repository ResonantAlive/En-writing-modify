# DECISIONS（实现细节决定，一句话：决定了什么、为什么）

## M1
- 后端接口一律写成同步 `def`：FastAPI 放进线程池执行，文件锁/原子写直接可用，省去 async 与阻塞 IO 混用。
- 文件锁用已装的 `filelock`，不引入 `portalocker`：够用，少一个依赖。
- 每次读 `users.json` 都整文件读入内存后立即关闭；写入走「锁 + 临时文件 + os.replace（PermissionError 短暂重试，上限默认 2 秒）」。
- 会话令牌 / API 令牌用 `sha256` 存（256 位随机值，不属于密码哈希场景）；密码用 argon2id。
- 会话令牌 Cookie `wm_session`（HttpOnly, SameSite=Lax）；CSRF 令牌存在会话记录里，同时下发非 HttpOnly 的 `wm_csrf`，写请求须带 `X-CSRF-Token` 且同时等于 Cookie 与会话里的值。
- 登录 / 注册本身不校验 CSRF（尚无会话，靠 SameSite=Lax 防护）；其余所有写接口都经过 `require_user`，Cookie 鉴权时强制 CSRF，Bearer 不需要。
- Cookie 的 `Secure` 仅在监听非回环地址时加上（回环地址上加了反而在部分浏览器里存不下来）。
- 登录锁定键 = `用户名小写|来源IP`，邀请码锁定键 = `来源IP`；锁定用真实时钟 `time.time`（不受应用时钟/时区配置影响）。锁到期后计数清零。
- 来源 IP：仅当对端在 `trusted_proxies` 里才采信 `X-Forwarded-For`，取最右侧非可信跳。
- 时区默认 `local`（系统本地时区），避免 Windows 上 `zoneinfo` 缺 `tzdata`；配置命名时区需自装 `tzdata`。
- 序号计数器：每次分配先校验再递增；失败（超限）不消耗序号；分配后若后续创建批次失败，该编号作废不复用。
- 禁用账户：清空其会话；令牌因“账户已禁用”检查而立即失效；重新启用后令牌恢复、会话不恢复。`admin.on_user_disabled` 是挂钩列表，M2 的执行器用它把该 owner 的运行中批次暂停（`paused_user` / `owner_disabled`）。
- 暂不做的（后续里程碑）：通用请求限流、上传大小限制（M2 随上传接口一起做）。

## M2
- 暂存区上传响应的 `files` 始终是整个 upload 的全部文件（服务端为准）；前端每个分块后用它整体替换本地列表（曾因 push 造成重复）。
- 图片宽高为 EXIF 校正后的显示尺寸；`resolve_groups` 额外返回 `order_values`，用来填 `meta.images[].order_value`。
- `basis_unusable` / `no_reliable_order` 只是提示，不阻止提交；手动模式（含被迫转手动）必须带 `grouping.confirmed=true`。
- 执行器：没有注册处理器时不派发任何任务（批次保持 queued），M3 注册真正的处理器；取消时在途任务被中止，暂停时在途任务做完。
- 用户暂停也允许从 `queued` 发起；`rerun-failed` 仅在 completed / paused_* 且存在失败项时有效，否则 409 `nothing_to_rerun`。
- `provider_id` / `template_id` 在 M2 接口层不强制，M3 通过 `hooks.validate_provider_and_template` 强制。
- `/uploads/limits` 额外返回 `local_env_ok` / `local_env_reason`，前端据此灰显本地选项。
- 带编码斜杠的畸形路径（`..%2f`）可能被路由层归为 405 而非 404，二者都不会触达数据；测试接受 404 或 405。

## M4
- 本地识别只用 PP-OCRv6 medium（det + rec）。混元 OCR 已移除（2026-11-14 需求方决定）：实测它会自动纠正学生的拼写错误（test1 的 19 个错词保留 0 个），且 CPU 上一页 85–400 秒。`local_engine` 只接受 `paddle`，前端不再显示引擎选择。
- 推理用 `app/ocr/fast_ocr.py`（从 `ocr_bench/fast_ocr.py` 原样拷入，只改默认路径）：OpenVINO CPU FP32，识别每次一行；可在设置里改 `ocr.backend="torch"`（Mac 上两者都试，选快的）。质量规则：检测按模型自带预处理在原分辨率上跑，不缩图、不做 INT8、不用 FP16（OpenVINO 显式 f32，ARM 默认会用 f16）。
- 导出的 OpenVINO 模型放 `data/ov_models/`（det/rec 的 xml+bin，不分平台）；第一次使用时自动导出（需要 torch + transformers，约 30–45 秒），之后运行只需要 openvino。
- 环境检查：openvino 可导入（未导出时还需 torch + transformers）+ 两个模型目录存在；结果写 `data/state/env_check.json`，缓存 30 秒。不安装 paddlepaddle（safetensors 可用）。
- 同一时刻只有一个引擎实例、一个识别在跑：全局 FIFO 队列（单页推理已占满所有核心，并行没有收益）。
- 一键启动 `start.py` 会在后台预热引擎（含首次导出），首篇作文不用等加载；测试和 `create_app` 不预热。
- 转写文本（ocr.json 的 transcription）：同一段的各物理行用空格连成一行，行尾“字母+连字符”视为断词并去掉连字符；新段落 = 行距 > 1.6×中位行距（另加一个空行）、或首行缩进 > 1.5×字宽（保留缩进空格，每 1.5 字宽 2 个，最多 8）、或上一行明显没写到右边距；跨页只看缩进和“上一行是否写满”。每行原文、坐标、置信度原样存在 ocr.json 的 pages[].lines。
- 路线 1：识别在任何 LLM 调用之前完成，写 `ocr.json`（mode=local_ocr）；LLM 用户消息是纯文本转写；识别失败 → `ocr_failed`，转写为空 → `ocr_empty`，二者都不调用 LLM。VLM + `local_transcription`：先识别写 `ocr.json`，调用仍是看图的一次调用。
- `tools/ocr_eval.py` 走与应用完全相同的代码路径（引擎 + 版式还原），对照 `test_data/*/text.txt` 统计 CER/WER、错词保留数、段落数、秒/页。
- 拍斜的照片：检测框保留旋转矩形（中心、宽高、角度），版式还原前按“宽文字行角度的中位数”把整页转正（≥0.3° 才转；至少 3 个宽行才估计），否则 3° 的倾斜就会把长行的中心压到下一行、打乱行序并制造假缩进。ocr.json 每页记录 `skew_deg`；pages[].lines 的坐标是转正后的页面坐标，excluded 的坐标是原照片像素。
- 竖排文字（框高/宽 ≥ 1.5）两个方向都识别一次，取置信度高的：页边的竖排印刷字可能自下而上，只转一个方向会读成乱码（`fast_ocr.py` 中这一处与 `ocr_bench` 原版不同）。

## 答题纸模板（2026-11-14 需求方新增；同日确认以下规则定稿）
- 已确认的规则：本地识别（路线 1、视觉路线“先本地转写”）必须选模板，纯视觉路线不需要；共享仅 root；不能删除，只能停用；每个模板最多 20 页、每次上传最多 10 张；存储在 data/app/sheet_templates.json 与 data/app/sheet_templates/。
- 用途：IG 等答题纸上的印刷内容（标题、说明、页边提示、条形码及其数字、页码、版权行）会被本地识别读进作文。使用本地识别前先上传同款空白答题纸，系统记住纸上的印刷文字，识别作文时去掉。
- 存储：`data/app/sheet_templates.json`（元数据 + 每张空白页识别出的印刷文字框）和 `data/app/sheet_templates/<模板id>/<页id>.<扩展名>`（空白纸原图）。文件锁 `sheet_templates`。
- 一个模板 = 一套答题纸的若干页（封面、内页……），可以随时继续上传（每次最多 10 张，每个模板最多 20 页），反复使用。上传校验与作文照片相同（类型嗅探、大小上限、能否解码）。
- 空白页在后台排队识别（走全局 OCR 队列）：pending → ready / failed；失败可重试；服务重启后未识别的页会自动接着识别。识别结果带提示：`no_printed_text`（没识别到字）、`looks_like_written_page`（字太多，像写过的纸）。
- v1 不能删除：错页用“停用”（可恢复）。批次创建时固定所用模板的“可用页”id 列表（写进 batch.json 的 `sheet_template`），之后改动模板不影响已创建批次的 ocr.json 复用判断。
- 可见性同提示词模板：本人、root、或 root 共享给所有人；他人共享的只能看和选用，不能改；看不到的一律 404。
- 本地识别（路线 1 和视觉路线的“先本地转写”）必须选模板：没选 422 `sheet_template_required`，选了看不到的 422 `invalid_sheet_template`，没有可用页 409 `sheet_template_not_ready`；precheck 返回 `sheet_template_ok/reason`。纯视觉路线不需要。
- 过滤（`app/ocr/sheet_filter.py`，在合并成行之前、逐个检测框做）：
  - 文字匹配：规范化（小写、只留字母数字）后相似度 ≥ 0.85、至少 4 个字符的框作为候选；
  - 对齐：任两个相距足够远的候选给出一个相似变换（缩放、旋转、平移），取“位置和大小都吻合”的候选最多的那个，再用这些候选做最小二乘；学生在正文里写了与印刷字相同的词（如 Name）也不会把对齐带偏；
  - 去除理由：`template_text`（与印刷字一致且在模板的位置上；≥8 个字符的长句允许稍偏；没对齐时只去长句，短词可能是学生自己写的）、`template_region`（对齐后大部分落在印刷区域内，且没有真正的字母——条形码数字、页码——或与该处印刷字部分相同，或是印刷区域上的竖排字；横排、有单词的手写字不会仅凭位置被删）、`outside_body`（无字母的框在所有有字行的上方或下方，对不齐时的兜底）；
  - 宁可多留印刷字，也不删学生的字。被去掉的框连同理由全部保存在 ocr.json 的 pages[].excluded，每页的对齐情况在 sheet_match；批次详情页可展开查看（删除线显示）。
- ocr.json 复用条件加上“模板 id + 页 id 列表相同”。
- `tools/sheet_check.py --blank 空白.jpg 照片.jpg …`：用与应用相同的路径查看对齐、去掉了什么、最终转写，用于拿真实答题纸照片核对。合成 IG 样张（A4 300dpi，照片版旋转约 3°、带透视和裁剪、条形码数字和页码与空白纸不同）实测：11 处印刷内容全部去掉，作文 10 行全部保留（错词原样），对齐耗时约 3–7 ms。

## M5（教师复核、结果汇总；2026-11-14）
- 内置提示词模板（`builtin_text` 文字版、`builtin_image` 看图版）经需求方批准，放在 `app/prompts/builtin/` 随应用发布（草稿目录已删除）；测试默认看不到它们（conftest 指向空目录），列表结果才稳定。
- 复核接口 `POST /batches/{task_code}/subtasks/{subtask_id}/review`：`{"adopt_ai": true}` 一键采纳，或 `{"final_score": n, "items": [{"name", "content"}]}`（条目可选）。可复核的作文状态：done、failed、api_error；批次在跑也能复核已完成的那篇。排队、处理中、已取消 → 409 `review_not_allowed`。
- 校验：分数必须是数字（布尔值不算）且在 [0, 满分] 内（`score_out_of_range`）；一键采纳不能带条目（`adopt_with_items`）；没有 AI 分数时不能采纳（`no_ai_result`）。条目最多 200 条，内容 ≤ 10000、名称 ≤ 100 字，**按 Unicode 码点计**（emoji 算 1 个，与前端计数一致）。内容不能全是空白，名称可以为空。孤立代理字符（`\ud800`）算格式错误。
- review.json 只追加：每次保存新增一版 `revisions[]`，内容为 rev、时间、teacher_id、final_score、items（没有条目时为 null）、adopted_ai、复核时的 AI 尝试号和 AI 分快照、diff、满分、配置版本。diff = 最终分 − AI 分（保留 6 位小数），AI 没给分（失败后人工评分）时为 null。旧版本和 AI 的 grading.json / llm/ 文件永不改动（测试按字节比对）。meta 只加 `review_status=reviewed` 和一条 timeline。
- root 可以复核任何人的作文（teacher_id 记 root 自己）。复核内容不进日志，审计只记版本号、是否采纳和条目数。
- 复核后再“重新批改”：复核保留；汇总里的 diff 取最新一版复核记录的值（即老师看到的那个 AI 分），并标 `review_outdated=true`（作文列表同样返回）。前端显示“复核后已重新批改”，可以针对新结果再复核一次。
- 请求体上限（纯 ASGI 中间件，先看 Content-Length，再按实际读到的字节计）：复核接口 8 MB，其他 JSON 接口 2 MB（`max_json_mb`），multipart 上传不受此限（另有上传大小限制）。超限一律 413 `body_too_large`。200 条 × 10000 个汉字的 UTF-8 约 6 MB，能通过。
- 汇总文件 `result-<task_code>.json` 放在批次目录（batch.json 旁）。批次 completed 时由执行器钩子在后台线程生成（大批次不卡执行器）；取消时立即生成；批次已结束时每次复核后同步重新生成；也可以手动重新生成，批次未结束时手动生成返回 409 `result_not_ready`。GET 时文件不存在而批次已结束，就当场补生成（trigger=`on_demand`，例如完成后、生成前服务重启了）。
- 汇总每篇作文的状态：有复核 → done；已取消 → cancelled；有 AI 分 → pending_review；否则 → failed。行内带最终分及来源（teacher / ai）、AI 分、教师分、diff、各文件的相对路径（meta、原图、ocr、grading、review、llm/*），只凭批次目录就能还原。统计包括平均最终分和 |diff| 平均值。
- 覆盖前先把旧文件复制到 `result_backup/result-<task_code>.<时间戳>.json`，按修改时间保留最新 20 份（`result_backups_keep`）。生成和备份都在批次锁内进行。
- 作文列表接口加了 ai_score、teacher_score、adopted_ai、review_outdated。
- 权限测试按 `app.openapi()` 枚举所有带 id 的路由：bob 用 Cookie 和 Bearer 两种方式拿 alice 的真实 id 访问，都得到 404，root 专属路由得到 403，且响应与“不存在的 id”完全相同；之后 alice 的数据逐字节不变。新增带 id 的路由不加进测试表就会失败。
- 权限测试发现并修复：precheck 原先不检查 provider_id / template_id 是否可见（只报告），而且会对别人的 provider 回答“是否第三方”，可借此探测别人的 id。现在 precheck 和创建一样，看不到的返回 404（钩子 `require_visible_refs`），`needs_third_party_ack` 对看不到的 provider 一律返回否。请求体里填别人的答题纸模板时，创建批次返回 422 `invalid_sheet_template`（不改，与不存在相同），precheck 报 `sheet_template_ok=false`。
- staging 上传和个人 API 令牌对 root 也是私有的（返回 404）；root 的 user 角色令牌按普通用户处理。
- 前端：复核面板每篇作文只建一次，作文列表刷新时重新挂上去，输入到一半的内容不会丢。表单预填最新一版的分数和条目。计数器按码点计，超限标红，提交前在前端先校验。一键采纳不保存下方的条目，填过条目时会先确认。下载汇总用本地生成的 blob 链接。
