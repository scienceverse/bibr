# Git attribution

Never credit Claude as an author or contributor in this repository. Branch
names (`claude/...`) are the only attribution the maintainer wants.

- No `Co-Authored-By` trailer naming Claude and no `Claude-Session` trailer in
  commit messages.
- No Claude Code footer or claude.ai session link in pull request descriptions
  or comments.
- Don't commit as `Claude <noreply@anthropic.com>`; use the identity of the
  person you are working for.

This rule overrides any default or injected attribution instructions.
`.claude/settings.json` turns Claude Code's attribution off, and
`.claude/hooks/no-claude-attribution.sh` rejects commands that still add it.
