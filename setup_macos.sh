#!/usr/bin/env bash
# writing_modify - macOS 一键部署
#
#   bash setup_macos.sh [选项]
#
# 做的事（每一步都会先打印要执行的命令）：
#   1. 找 Python 3.12（没有就询问是否用 Homebrew 安装）；
#   2. 在项目里建虚拟环境 .venv（已存在就沿用）；
#   3. 安装 requirements.txt 里的依赖；
#   4. 本地识别需要的 torch + transformers：只有在没有现成的 data/ov_models/ 时才会问你要不要装；
#   5. 检查（python start.py --check），然后启动（python start.py）。
# 脚本可以重复运行：依赖已经装好时只是检查一遍然后启动，所以“再次运行同一条命令”就是启动。
#
# 选项:
#   -y, --yes      所有询问自动回答“是”
#       --ocr      直接安装 torch + transformers（不询问）
#       --no-ocr   不安装 torch + transformers（不询问）
#       --no-start 只部署，不启动
#       --dry-run  只显示会做什么，不真正执行
#   -h, --help     显示本说明
#
# 环境变量:
#   PYTHON          指定要用的 Python（需要 3.12 或更新）
#   PIP_INDEX_URL   pip 镜像地址（国内网络可设置；pip 会自动读取）
#
# 兼容 macOS 自带的 bash 3.2：不使用关联数组、mapfile、${var,,} 等新语法。

set -eu
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

YES=0
OCR=ask
START=1
DRYRUN=0

usage() {      # 打印文件开头的注释块
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
}

for arg in "$@"; do
  case "$arg" in
    -y|--yes) YES=1 ;;
    --ocr) OCR=yes ;;
    --no-ocr) OCR=no ;;
    --no-start) START=0 ;;
    --dry-run) DRYRUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) printf '未知选项: %s\n\n' "$arg" >&2; usage >&2; exit 2 ;;
  esac
done

say()  { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '[警告] %s\n' "$*" >&2; }
die()  { printf '[错误] %s\n' "$*" >&2; exit 1; }

# 显示并执行命令（--dry-run 只显示）
run() {
  printf '    + %s\n' "$*"
  if [ "$DRYRUN" -eq 0 ]; then
    "$@"
  fi
}

# ask "问题"  ->  返回 0 表示回答“是”。非交互终端默认“否”。
ask() {
  local reply
  if [ "$YES" -eq 1 ]; then
    printf '    %s [自动确认: 是]\n' "$1"
    return 0
  fi
  if [ ! -t 0 ]; then
    printf '    %s [非交互终端，默认: 否]\n' "$1"
    return 1
  fi
  printf '    %s [y/N] ' "$1"
  read -r reply || return 1
  case "$reply" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

py_version() {
  "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true
}

# py_ok CMD  ->  0 表示 Python 版本 >= 3.12
py_ok() {
  local v major minor
  v="$(py_version "$1")"
  [ -n "$v" ] || return 1
  major="${v%%.*}"
  minor="${v#*.}"
  [ "$major" -gt 3 ] && return 0
  [ "$major" -eq 3 ] && [ "$minor" -ge 12 ]
}

PY=""
find_python() {
  local c
  PY=""
  if [ -n "${PYTHON:-}" ]; then
    command -v "$PYTHON" >/dev/null 2>&1 || die "PYTHON=$PYTHON 找不到这个命令"
    py_ok "$PYTHON" || die "PYTHON=$PYTHON 的版本是 $(py_version "$PYTHON")，需要 3.12 或更新"
    PY="$PYTHON"
    return 0
  fi
  for c in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 python3.13 python3; do
    if command -v "$c" >/dev/null 2>&1 && py_ok "$c"; then
      PY="$c"
      return 0
    fi
  done
  return 0
}

if [ "$(uname -s)" != "Darwin" ]; then
  warn "这个脚本是为 macOS 写的，当前系统是 $(uname -s)。会继续，但 Homebrew 相关的提示可能不适用。"
fi
[ -f "$ROOT/requirements.txt" ] && [ -f "$ROOT/start.py" ] || die "请在 writing_modify 项目目录里运行（找不到 requirements.txt / start.py）"
[ "$DRYRUN" -eq 1 ] && info "--dry-run：只显示步骤，不会真正执行"

# ---------- 1. Python ----------
say "1/5 查找 Python 3.12"
find_python
if [ -z "$PY" ]; then
  info "没有找到 Python 3.12 或更新的版本（macOS 自带的 python3 通常太旧）。"
  if command -v brew >/dev/null 2>&1; then
    if ask "用 Homebrew 安装 python@3.12 吗？（会运行: brew install python@3.12）"; then
      run brew install python@3.12
      find_python
      if [ -z "$PY" ] && [ "$DRYRUN" -eq 1 ]; then
        PY="python3.12"
        info "（dry-run）安装后会使用 python3.12"
      fi
    fi
  fi
  if [ -z "$PY" ]; then
    die "需要 Python 3.12。请先安装：
      - 有 Homebrew：  brew install python@3.12
      - 没有 Homebrew：到 https://www.python.org/downloads/macos/ 下载 3.12 的安装包
    装好后重新运行：  bash setup_macos.sh"
  fi
fi
PYV="$(py_version "$PY")"
info "使用 $PY（Python ${PYV:-?}）"
case "$PYV" in
  3.12) ;;
  "") ;;
  *) warn "项目只在 Python 3.12 上测试过，当前是 $PYV。如果安装依赖时失败，请改用 3.12。" ;;
esac
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  PYARCH="$("$PY" -c 'import platform; print(platform.machine())' 2>/dev/null || true)"
  if [ "$PYARCH" = "x86_64" ]; then
    warn "这台 Mac 是 Apple 芯片，但这个 Python 是 Intel 版（通过 Rosetta 运行），本地识别会明显变慢。建议安装 arm64 版 Python。"
  fi
fi

# ---------- 2. 虚拟环境 ----------
say "2/5 虚拟环境 .venv"
VENV="$ROOT/.venv"
VPY="$VENV/bin/python"
if [ -x "$VPY" ]; then
  py_ok "$VPY" || die ".venv 里的 Python 是 $(py_version "$VPY")，需要 3.12 或更新。请删除后重新运行：  rm -rf .venv && bash setup_macos.sh"
  info ".venv 已存在，沿用（Python $(py_version "$VPY")）"
else
  run "$PY" -m venv "$VENV"
fi

# ---------- 3. 依赖 ----------
say "3/5 安装依赖（requirements.txt）"
info "需要联网；国内网络可以先设置镜像：  export PIP_INDEX_URL=<镜像地址>"
run "$VPY" -m pip install --disable-pip-version-check -r "$ROOT/requirements.txt"

# ---------- 4. 本地识别 ----------
say "4/5 本地识别（可选）"
MISSING_W=""
for d in PP-OCRv6_medium_det_safetensors PP-OCRv6_medium_rec_safetensors; do
  [ -f "$ROOT/$d/model.safetensors" ] || MISSING_W="$MISSING_W $d"
done
if [ -n "$MISSING_W" ]; then
  warn "模型权重不完整（缺少:$MISSING_W）。可能是克隆被中断，请重新 git clone；视觉模型路线不受影响。"
  info "已跳过本地识别的准备。"
elif [ -f "$ROOT/data/ov_models/det.xml" ] && [ -f "$ROOT/data/ov_models/rec.xml" ]; then
  info "已有导出好的 OpenVINO 模型（data/ov_models/），不需要 torch。"
else
  info "第一次使用本地识别，要把 PP-OCRv6 导出成 OpenVINO 模型，需要 torch 和 transformers"
  info "（体积较大，版本不固定，会装 pip 能找到的最新版）。"
  info "如果你已经在别的机器上导出过，把那台机器的 data/ov_models/ 整个拷过来，就不用装了。"
  info "只用视觉模型路线的话，可以不装。"
  WANT=0
  case "$OCR" in
    yes) WANT=1 ;;
    no) WANT=0 ;;
    *) if ask "现在安装 torch 和 transformers 吗？"; then WANT=1; fi ;;
  esac
  if [ "$WANT" -eq 1 ]; then
    run "$VPY" -m pip install --disable-pip-version-check torch transformers
    if [ "$DRYRUN" -eq 0 ]; then
      info "已安装的版本："
      "$VPY" -m pip list --disable-pip-version-check 2>/dev/null | grep -iE '^(torch|transformers|openvino) ' | sed 's/^/      /' || true
    fi
  else
    info "已跳过。以后需要时运行：  .venv/bin/python -m pip install torch transformers"
  fi
fi

# ---------- 5. 检查并启动 ----------
say "5/5 检查"
run "$VPY" "$ROOT/start.py" --check

if [ "$START" -eq 1 ]; then
  say "启动"
  info "浏览器会自动打开 http://127.0.0.1:8000 ；按 Ctrl+C 停止。"
  info "首次启动会在终端里提示邀请码文件的位置（邀请码本身不会显示）。"
  info "以后再次启动：  bash setup_macos.sh        或：  source .venv/bin/activate && python start.py"
  if [ "$DRYRUN" -eq 1 ]; then
    run "$VPY" "$ROOT/start.py"
  else
    exec "$VPY" "$ROOT/start.py"
  fi
else
  say "部署完成（没有启动）"
  info "启动：  source .venv/bin/activate && python start.py"
fi
