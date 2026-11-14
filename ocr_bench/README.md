# OCR 推理速度测试（CPU / FP32 / 不降分辨率）

独立测试文件夹，不改动仓库其他文件。环境：`D:\Transformer\.venv`（torch 2.12 CPU、transformers 5.23、OpenVINO 2026.2），
Intel Core Ultra 5 325（4P+4E，8 线程）。测试数据：`../test_data`，共 5 篇作文、8 页。

## 约束

- 只提高硬件利用率（MFU），**不做降低质量的处理**：检测按模型自带配置预处理（limit_type=min 736 / max 4000，
  取整到 32 的倍数，这批 A4 扫描页 = 原分辨率 2112×2976）；只用 FP32（OpenVINO 的模型文件也存 FP32 权重）；不做 INT8，不用缩略图。
- 质量校验：`verify_equiv.py` 在真实页面上比对各后端和 torch FP32 的输出（检测概率图最大偏差约 3e-5，识别 argmax 全部相同）；
  再对照 `text.txt` 算 WER/CER 和"错词保留数"（作文里的拼写错误必须原样识别出来）。

## CPU 峰值和模型算力需求（`mfu.py`）

- 实测 sgemm 峰值：8 线程约 410–435 GFLOPS（单个 P 核约 106）。下文 MFU = 实际 GFLOPS / 这个峰值。
- 原分辨率检测每页约 **2.7 TFLOP**；识别每行（宽 1024）约 **36 GFLOP**，算量和宽度成正比。

## 内核 MFU（`cpu_mfu_bench.py`，8 线程）

| 后端 | 检测（2112×2976） | 识别（batch 1 → 28） |
|---|---|---|
| torch eager（现状） | 56% | 18% → 42% |
| torch channels_last | **65%** | 18% → 43% |
| torch jit freeze | 59% | 35% → 61% |
| **OpenVINO CPU FP32（LATENCY）** | **77%** | **59% → 67%** |

4 线程（只用 P 核）在 torch 下更慢，E 核有贡献。OpenVINO 的 THROUGHPUT 模式在单请求时只用到部分核心，不适合单页处理。

## 完整流水线（8 页，`ppocr_cpu.py`；`results/SUMMARY.md`）

| 方案 | 秒/页 | 检测 | 识别 | 前后处理 | WER | CER | 错词保留 |
|---|---|---|---|---|---|---|---|
| transformers 原样流程（原分辨率，基线） | 14.88 | 10.19 | 4.25 | 0.43 | 0.0916 | 0.0122 | 54/65 |
| 本流水线 + torch eager | 15.18 | 10.07 | 4.74 | 0.38 | 0.0876 | 0.0112 | 56/65 |
| 本流水线 + torch 最优组合（检测 channels_last，识别 jit） | 12.56 | 9.22 | 2.95 | 0.39 | 0.0883 | 0.0116 | 56/65 |
| **本流水线 + OpenVINO FP32，识别 batch=1（推荐）** | **9.59** | 7.05 | 2.12 | 0.41 | **0.0828** | **0.0107** | 55/65 |
| 本流水线 + OpenVINO FP32，识别 batch=8 | 9.16 | 6.67 | 2.10 | 0.38 | 0.0876 | 0.0112 | 56/65 |
| OpenVINO，2 页并行（2 streams） | 10.20 | — | — | — | 0.0883 | 0.0116 | 56/65 |

- 这台笔记本有约 ±10% 的功耗/温度波动（同一检测配置两次分别是 6.67s 和 7.47s），batch 1 和 batch 8 的差距在噪声范围内。
- 推荐识别用 batch=1：在 OpenVINO 上几乎不增加耗时，每行结果独立，不受同批其他行 padding 的影响，WER/CER 最好。
- 理论下限（100% sgemm 峰值）大约是 7.8 秒/页；现在约 9.2–9.6 秒，已经到下限的 80% 左右。
- Python 侧前后处理只占约 0.4 秒/页（4%），所以**同样的内核换成 C++，PP-OCR 最多再快约 5%**。

## 混元 OCR（`bench_hunyuan.py`，test1 照片页，FP32，模型自带预处理）

| 配置 | 总耗时 | 视觉编码 | 预填充 | 解码 |
|---|---|---|---|---|
| eager 注意力（app 现用） | 159.8 s | 103.6 s | 10.3 s | 45.8 s（5.8 tok/s） |
| **SDPA 注意力** | **85.2 s** | 41.8 s | 6.4 s | 36.9 s（7.2 tok/s） |
| SDPA，解码阶段改 4 线程 | 89.8 s | 42.3 s | 6.4 s | 41.0 s（6.5 tok/s） |

- SDPA 和 eager 输出逐字相同，SDPA 快 1.9 倍。视觉编码在 SDPA 下约 92% MFU（约 15.9 TFLOP / 41.8 s），已经接近上限。
- 解码瓶颈在 HF `generate` 每步的框架开销：每 token 读约 2.6 GB（权重 + KV），只用到约 19 GB/s 带宽。
  这部分换专用运行时（C++ / 有状态 KV 的推理引擎）提升空间最大。
- **质量问题：混元会自动纠正学生的拼写和语法错误**（goverment→government、create→creates、have→has），test1 的 19 个错词保留了 0 个。
  用于作文批改的错词检查不合适。

## 直接使用：`fast_ocr.py`

推荐方案（OpenVINO CPU FP32，识别每次一行）的单文件版本，可以单独拷走。运行时只依赖 numpy、Pillow、openvino。
torch 和 transformers 只在第一次导出模型时需要；导出的 `cache/det.*`、`cache/rec.*` 不分平台，可以直接拷到别的机器上用。

```python
from fast_ocr import FastPPOCR
ocr = FastPPOCR()                         # 首次导出模型到 cache/（约 30–45 秒），之后加载不到 1 秒
ocr.warmup()                              # 服务启动时调用一次，首页少等约 2–3 秒
text = ocr.recognize_text("page_1.png")   # 整页文字
lines = ocr.recognize("page_1.png")       # [Line(text, box, score)]，按阅读顺序
essay = ocr.recognize_pages(["p1.png", "p2.png"])   # 多页拼成一篇
```

已验证：8 页结果和基准测试逐行一致；`backend="torch"` 的结果也和 OpenVINO 一致。

## 文件

- `fast_ocr.py`：推荐方案的单文件版本（实际使用就用它）
- `common.py`：测试数据读取、指标、numpy 实现的 DB 后处理（游程连通域 + 最小外接矩形，替代逐像素 Python BFS）、裁剪、CTC 解码
- `engines.py`：torch / torch-jit / OpenVINO 后端，以及 IR 导出（`cache/`）
- `ppocr_cpu.py`：优化后的流水线
- `bench_hf_baseline.py`：transformers 原样流程基线
- `cpu_mfu_bench.py`、`mfu.py`：内核 MFU 和 FLOPs
- `verify_equiv.py`：各后端与 torch FP32 的输出一致性
- `bench_hunyuan.py`：混元耗时拆分
- `report.py` → `results/SUMMARY.md`

## 复现

```bash
cd D:\Transformer\writing_modify\ocr_bench
D:\Transformer\.venv\Scripts\python.exe ppocr_cpu.py --backend ov --rec-bs 1
D:\Transformer\.venv\Scripts\python.exe bench_hf_baseline.py --det-max-side 0 --runs 1
D:\Transformer\.venv\Scripts\python.exe verify_equiv.py torch-cl torch-jit ov
D:\Transformer\.venv\Scripts\python.exe bench_hunyuan.py --attn sdpa --pages test1/0
D:\Transformer\.venv\Scripts\python.exe report.py
```
