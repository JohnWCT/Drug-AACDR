#!/usr/bin/env bash
# AACDR Docker 工作流程測試
#
# 用途：在已啟動的 AACDR 容器內驗證
#   (A) Git 倉庫可讀（volume mount + safe.directory）
#   (B) AACDR 程式流程可從資料準備跑到訓練
#
# 使用方式（於本機執行，勿進入容器手動操作）：
#   docker exec AACDR bash /workspace/AACDR/scripts/docker_workflow_test.sh
#
# 設計原則：
# - 所有設定僅寫入容器內（如 git safe.directory），不修改本機 git config
# - 工作目錄固定為 /workspace/AACDR（與 docker run -w 一致）
# - 任一步驟失敗即 exit 1，便於 CI / 手動排查

set -euo pipefail

PROJECT_ROOT="/workspace/AACDR"
cd "${PROJECT_ROOT}"

echo "========================================"
echo " Step 0: 環境檢查"
echo "========================================"
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
test -d "${PROJECT_ROOT}/code"
test -d "${PROJECT_ROOT}/data"
echo "[PASS] project layout"

echo
echo "========================================"
echo " Step 1: Git 流程（容器內）"
echo "========================================"
# root 使用者掛載 uid=1001 的 .git 時，Git 2.35+ 會拒絕操作。
# 僅在容器全域設定 safe.directory，不影響本機。
git config --global --add safe.directory "${PROJECT_ROOT}"

git status --short --branch
git log -1 --oneline
git remote -v
echo "[PASS] git readable inside container"

echo
echo "========================================"
echo " Step 2: AACDR 煙霧測試"
echo "========================================"
python "${PROJECT_ROOT}/scripts/aacdr_smoke_test.py"

echo
echo "========================================"
echo " ALL WORKFLOW TESTS PASSED"
echo "========================================"
