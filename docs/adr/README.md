# Architecture decision records

Write an ADR only when all of these are true:

- another reasonable choice existed;
- reversal is costly or the question is likely to recur;
- the decision cannot be recovered from code;
- it crosses more than one domain;
- `AGENTS.md` routes the responsible agent to it.

Test the boundary by asking: **would violating this be a bug?** If not, it is
history and Git already records it.

Implementation plans expire when shipped. Investigation reports belong here
only after their durable decisions have been reduced to a contract. A
single-domain contract belongs in its responsible skill, not in this directory.
