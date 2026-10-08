# Engineering workflow

## Language

- Project prose language: English

AI-only instructions, schemas, identifiers, template headings, tool keywords,
and canonical terms remain in English. Human-reviewed repository documentation,
ADRs, commit messages, and pull requests also use English to match the existing
project. Conversation with the user may use the user's language.

## Authorization and delivery

The root `AGENTS.md` ("How a change is made") owns how a change is designed,
approved, implemented, reviewed, and delivered. In short: the maintainer
approves a written design; that approval covers commits, pushes, and the one
pull request for that design on its branch; the maintainer merges. Where a
generic workflow skill asks for separate commit, push, or pull-request approval,
the approved design already gives it; open pull requests ready for review, not
as drafts. Posting an issue or a comment, and anything on a repository the
maintainer does not own, still needs the maintainer's explicit instruction.

## Branch policy

- Default branch: `main`
- Work branches use a descriptive `fix/`, `feat/`, `docs/`, or `release/`
  prefix.
