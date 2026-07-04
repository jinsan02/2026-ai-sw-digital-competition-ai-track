#!/bin/bash
# Linux 구조+기능 스모크 (노트북 WSL). 버전은 평가서버와 다르니 구조 검증이 주목적.
set -e
ZIP=/mnt/c/dacon/submit_large384.zip
W=/tmp/smoke_l384
rm -rf "$W"; mkdir -p "$W"; cd "$W"
echo "=== unzip 엔트리 백슬래시 개수 (0이어야 Linux OK) ==="
unzip -l "$ZIP" | awk '{print $4}' | grep -c '\\' || echo 0
unzip -q "$ZIP"
echo "=== 추출 후 경로 확인 ==="
test -f model/hf_model/model.int8.safetensors && echo "  model.int8.safetensors OK"
test -f model/sparse_svc.pkl && echo "  sparse_svc.pkl OK"
test -f model/hf_meta.json && echo "  hf_meta.json OK"
test -f script.py && echo "  script.py OK"
mkdir -p data
cp /mnt/c/dacon/open/data/test.jsonl /mnt/c/dacon/open/data/sample_submission.csv data/
echo "=== script.py 실행 (CPU offline) ==="
source ~/dacon-venv/bin/activate
CUDA_VISIBLE_DEVICES="" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python script.py 2>&1 | grep -viE "some weights|should probably|futurewarning|warn" | tail -6
echo "=== output ==="
head -3 output/submission.csv
rows=$(($(wc -l < output/submission.csv) - 1))
echo "rows=$rows"
[ "$rows" -ge 1 ] && echo "LINUX SMOKE PASSED" || echo "LINUX SMOKE FAIL"
