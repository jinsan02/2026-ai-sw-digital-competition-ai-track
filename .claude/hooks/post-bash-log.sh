#!/bin/bash
# post-bash-log.sh — PostToolUse(Bash) : 감사 로그만 기록. 항상 exit 0.

INPUT=$(cat)
mkdir -p .claude/logs 2>/dev/null
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
CMD=$(printf '%s' "$INPUT" | sed -n 's/.*"command"[[:space:]]*:[[:space:]]*"\(.*\)".*/\1/p' | head -1)
echo "{\"ts\":\"$TS\",\"command\":\"$(printf '%s' "$CMD" | head -c 200)\"}" >> .claude/logs/audit.jsonl
exit 0
