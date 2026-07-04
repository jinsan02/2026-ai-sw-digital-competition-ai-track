#!/bin/bash
# pre-bash-guard.sh — PreToolUse(Bash) : 진짜 위험한 패턴만 차단.
# 설계 원칙: default-allow. stdin JSON 파싱 실패/미매칭이면 무조건 허용(exit 0) → 세션 브릭 방지.
# 차단 시 exit 2 (Claude Code에 stderr 피드백).

INPUT=$(cat)
mkdir -p .claude/logs 2>/dev/null
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# tool_input.command 추출 (jq 없이, 실패하면 빈 문자열 → 허용)
CMD=$(printf '%s' "$INPUT" | sed -n 's/.*"command"[[:space:]]*:[[:space:]]*"\(.*\)".*/\1/p' | head -1)
[ -z "$CMD" ] && exit 0

# 명백히 파괴적인 것만 (rm -rf 일반은 허용 — build/cache 정리에 흔함)
BLOCKED='rm[[:space:]]+-[rf]+[[:space:]]+/([[:space:]]|$)|:\(\)\{|mkfs|dd[[:space:]]+if=.*of=/dev|>[[:space:]]*/dev/sd|sudo[[:space:]]+rm|chmod[[:space:]]+-R[[:space:]]+777[[:space:]]+/'
if printf '%s' "$CMD" | grep -qiE "$BLOCKED"; then
  echo "{\"ts\":\"$TS\",\"action\":\"BLOCKED\",\"cmd\":\"$CMD\"}" >> .claude/logs/bash-guard.jsonl
  echo "[pre-bash-guard] 위험 패턴 차단: $CMD" >&2
  exit 2
fi

echo "{\"ts\":\"$TS\",\"action\":\"ALLOWED\",\"cmd\":\"$CMD\"}" >> .claude/logs/bash-guard.jsonl
exit 0
