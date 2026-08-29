# Engineering workflow

## Language

- Project prose language: English

AI-only instructions, schemas, identifiers, template headings, tool keywords,
and canonical terms remain in English. Human-reviewed repository documentation,
ADRs, commit messages, and pull requests also use English to match the existing
project. Conversation with the user may use the user's language.

## Authorization and delivery

1. Create a work branch before the first approved repository change. Do not
   implement directly on the default branch.
2. Require a specification for new features and behavior changes. Clear small
   fixes, behavior-preserving refactors, documentation, and mechanical
   configuration changes may use an approved written plan instead.
3. Use tickets only when work spans multiple sessions, cannot fit safely in one
   context, or the user explicitly asks for one.
4. Domain-document approval authorizes only the approved glossary or ADR edit.
5. Specification and ticket publication each require explicit approval.
6. An explicit implementation approval authorizes code, tests, internal review,
   and safe fixes. It does not authorize a commit.
7. Use test-driven development for new behavior and bug fixes. Documentation,
   comments, generated mirrors, and behavior-preserving mechanical changes are
   exempt.
8. Review the complete uncommitted worktree against repository standards and
   the approved requirements before requesting a commit.
9. Behavior-changing commits require a separate read-only review. P0 and P1
   findings block the commit; re-review after fixing them.
10. Commit requires explicit approval after a commit packet is presented.
11. Push and pull-request creation require separate approval. Default to a draft
    pull request. Mark it ready only after CI succeeds and the user approves.
12. Posting an issue or pull-request comment requires the user's explicit
    instruction to post it. Merge remains a user action unless separately
    authorized.

## Branch policy

- Default branch: `main`
- Work branches use a descriptive `fix/`, `feat/`, `docs/`, or `release/`
  prefix.
- Check the current GitHub branch-protection settings before relying on a
  server-side review boundary. Repository workflow requirements still apply
  when branch protection is absent or less strict.
