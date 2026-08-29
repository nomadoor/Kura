# Domain documentation

Kura uses one repository-wide domain context.

- `AGENTS.md` owns repository operating rules and routes work to the relevant
  project skill.
- `docs/adr/` owns durable, cross-domain architectural decisions that satisfy
  the criteria in `docs/adr/README.md`.
- Focused documents under `docs/` own user-facing or subsystem-specific facts.
- `.agents/skills/` owns operational guidance for a single development or
  usage domain. `.claude/skills/` is generated and must not be edited directly.

Do not add parallel context maps or duplicate repository rules without clear
evidence that the single root context has become insufficient.
