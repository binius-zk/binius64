---
name: jj-describe
description: Describe the current commit in jujutsu (jj)
disable-model-invocation: true
allowed-tools: Bash(jj *)
---

## Current changes

Diff of the current commit:
!`jj diff`

## Recent commit history (for style reference)

!`jj log -n 10 --no-graph`

## Your task

Write a commit description for the current changes based on the diff above.

Follow the commit message conventions observed in the log:
- If commits use brackets for crate names (e.g., `[crate-name]`), follow that pattern
- Keep the first line under 72 characters
- Use imperative mood (e.g., "Add feature" not "Added feature")
- Focus on the "why" rather than the "what"

Set the description using: `jj describe -m "Your commit message"`
