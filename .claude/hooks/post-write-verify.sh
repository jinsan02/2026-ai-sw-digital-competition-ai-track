#!/bin/bash
# post-write-verify.sh — PostToolUse(Write) : .py 구문 체크 (정보성, 비차단).
# python 없거나 실패해도 exit 0 → 흐름 안 끊음. 구문오류만 stderr로 알림.

INPUT=$(cat)
FP=$(printf '%s' "$INPUT" | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
case "$FP" in *.py) ;; *) exit 0 ;; esac

# 동작하는 python 인터프리터 탐색 (Windows store 스텁 회피: 'pass' 실행으로 검증)
PY=""
for c in python python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "pass" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -z "$PY" ] && exit 0

OUT=$("$PY" -m py_compile "$FP" 2>&1)
if [ $? -ne 0 ]; then
  echo "[post-write-verify] Python 구문 오류: $FP" >&2
  printf '%s\n' "$OUT" | head -5 >&2
fi
exit 0
