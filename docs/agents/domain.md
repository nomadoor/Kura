# Domain documentation

Kura uses one repository-wide domain context.

- `AGENTS.md` owns repository operating rules and routes work to the relevant
  project skill.
- A root `CONTEXT.md` may be created lazily when Kura needs a durable canonical
  glossary. Do not introduce `CONTEXT-MAP.md` or multiple context glossaries
  without clear evidence that the repository contains genuinely independent
  bounded contexts.
- `docs/adr/` owns durable, cross-domain architectural decisions that satisfy
  the criteria in `docs/adr/README.md`.
- Focused documents under `docs/` own user-facing or subsystem-specific facts.
- `.agents/skills/` owns operational guidance for a single development or
  usage domain. `.claude/skills/` is generated and must not be edited directly.

Do not add parallel context maps or duplicate repository rules without clear
evidence that the single root context has become insufficient. Before domain
or architectural work, read the relevant glossary and ADRs, use canonical
terms in issues, specifications, tests, and code, and surface conflicts rather
than silently overriding them.
