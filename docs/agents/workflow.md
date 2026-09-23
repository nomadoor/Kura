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
2. Before specifying or implementing a new feature or behavior change in
   existing code, apply `prior-art`. Reuse a completed survey, and return to
   clarification when its findings change the premise or scope.
3. Require a specification for new features and behavior changes. Clear small
   fixes, behavior-preserving refactors, documentation, and mechanical
   configuration changes may use an approved written plan instead.
4. Use tickets when work spans multiple sessions or agents, cannot fit safely
   in one context window, or the user explicitly asks for one.
5. Domain-document approval authorizes only the approved glossary or ADR edit.
6. Specification and ticket publication each require explicit approval.
7. An explicit `GO` authorizes implementation, tests, internal review, and safe
   fixes. It does not authorize a commit.
8. Use test-driven development for new behavior and bug fixes. Documentation,
   comments, behavior-preserving mechanical changes, generated files, external
   configuration, and emergencies followed by regression-test work are exempt.
9. Review the complete uncommitted worktree against repository standards and
   the approved specification or requirements source before requesting a
   commit.
10. Behavior-changing commits require a separate, read-only AI review using the
    approved requirements source, any specification and tickets that exist,
    relevant domain documentation, the diff, and verification results.
11. Return review findings to the implementation session. P0 and P1 findings
    block the commit; re-review after fixing them.
12. Commit requires explicit approval after a commit packet is presented.
13. Push and pull-request creation require separate approval. Default to a draft
    pull request.
14. Mark a draft pull request ready only after CI succeeds and the user
    approves. Merge remains outside the initial workflow.
15. Posting an issue or pull-request comment requires the user's explicit
    instruction to post it.

Review findings use P0/P1/P2/P3 severity. P0 and P1 block commit.

## Branch policy

- Default branch: `main`
- Work branches use a descriptive `fix/`, `feat/`, `docs/`, or `release/`
  prefix.
- Check the current GitHub branch-protection settings before relying on a
  server-side review boundary. Repository workflow requirements still apply
  when branch protection is absent or less strict.
