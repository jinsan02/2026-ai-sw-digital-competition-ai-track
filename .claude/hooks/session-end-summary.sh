#!/bin/bash
# session-end-summary.sh — Stop : 세션 요약 출력·기록. 항상 exit 0.
# (원본의 "CLAUDE.md 자동 수정"은 위험해서 제거 — 로그만 남김)

mkdir -p .claude/logs 2>/dev/null
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
CHANGED=$(git diff --name-only HEAD 2>/dev/null | head -10 | tr '\n' ' ')
ALLOWED=$(grep -c '"ALLOWED"' .claude/logs/bash-guard.jsonl 2>/dev/null || echo 0)
BLOCKED=$(grep -c '"BLOCKED"' .claude/logs/bash-guard.jsonl 2>/dev/null || echo 0)

echo "{\"ts\":\"$TS\",\"changed\":\"$CHANGED\",\"allowed\":$ALLOWED,\"blocked\":$BLOCKED}" >> .claude/logs/sessions.jsonl
echo "[session-end] 변경파일: ${CHANGED:-없음} | 허용 $ALLOWED | 차단 $BLOCKED" >&2
exit 0
