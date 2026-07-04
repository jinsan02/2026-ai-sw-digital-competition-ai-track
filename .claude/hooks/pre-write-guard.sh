#!/bin/bash
# pre-write-guard.sh — PreToolUse(Write) : 보호 경로 쓰기 차단.
# default-allow. file_path 못 뽑으면 허용(exit 0). 매칭 시 exit 2.

INPUT=$(cat)
mkdir -p .claude/logs 2>/dev/null
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

FP=$(printf '%s' "$INPUT" | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
[ -z "$FP" ] && exit 0

# 보호: 운영데이터 / git 내부 / venv / 바이너리 가중치 (손으로 쓸 일 없음)
case "$FP" in
  *open/data/*|*open\\data\\*|*/.git/*|*\\.git\\*|*/venv/*|*\\venv\\*|*.pkl|*.pt|*.safetensors|*.bin)
    echo "{\"ts\":\"$TS\",\"action\":\"BLOCKED\",\"file\":\"$FP\"}" >> .claude/logs/write-guard.jsonl
    echo "[pre-write-guard] 보호 경로 쓰기 차단: $FP" >&2
    exit 2 ;;
esac
exit 0
