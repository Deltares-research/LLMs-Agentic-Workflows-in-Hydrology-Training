---
name: my-skill-name
description: One or two sentences saying WHAT this skill does and WHEN the agent should use it. The agent only sees this line until it decides to load the skill, so include the trigger words a user would use (e.g. "flow duration curve", "drought", "return period").
---

# Title

Short context the agent does not already know (domain rules, file locations, conventions).

## Steps

1. ...
2. ...

## Gotchas

- Things that went wrong before and how to avoid them.

## Optional: scripts

Put helper scripts in `scripts/` next to this file and tell the agent to *run* them
(`<python> .claude/skills/<name>/scripts/x.py ...`) rather than re-writing the logic.
