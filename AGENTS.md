# Instructions for coding agents

Read this file before changing this repository.

## Branch and release
- Start from the latest `lab` branch and submit every feature or fix PR to `lab`, never `main`.
- `main` changes only via a `lab` to `main` PR reviewed and merged by the owner.
- Refresh `lab` before editing `research/features.py` or `research/run.py`, since multiple agents work on these files.
- Do not deploy or publish models without an independent review.

## Checks before pushing
- Run `python -m compileall -q research` and `python -m unittest discover -s research -p 'test_*.py' -v`.
- Run applicable historical-data and model GitHub Actions checks. Do not declare a PR ready unless every required check is green.
- When touching Foxy Pro in bottomupapp/bottomup, follow its root AGENTS.md and execute all prescribed API, web, i18n and parity checks.

## Research integrity
- Preserve the sealed holdout. Development experiments may not read locked evaluation labels.
- Validate point-in-time availability, funding/metrics publication delays, missing minute masks, and realistic trading costs.
- Model hit rate is not realized profit. No live trading or unsupported performance claims.
- Keep Fishnet an independent product, not automatically integrated into Foxy Score.

## Writing and security
- Committed content, comments and messages are in English. Avoid em dashes.
- Use real newlines in source files, not literal backslash-n escape text where line breaks belong.
- Never commit credentials or API tokens.
