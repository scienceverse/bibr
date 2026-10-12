#!/usr/bin/env bash
# Blocks commits and PR descriptions that credit Claude (see .claude/CLAUDE.md).
# Reads the PreToolUse payload on stdin; exit 2 rejects the call and shows
# stderr to Claude so it can retry without the attribution.
input=$(cat)
pattern='co-authored-by:[^"]*(claude|anthropic)|claude-session:|generated with \[?claude code|claude\.ai/code/session_'
if printf '%s' "$input" | grep -qiE "$pattern"; then
  echo "Blocked: this repo does not credit Claude. Remove Co-Authored-By/Claude-Session trailers, the Claude Code footer and session links, then retry." >&2
  exit 2
fi
exit 0
