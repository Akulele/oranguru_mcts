# Data Directory

This directory mixes two kinds of files:
- small tracked reference data needed by the bot at runtime
- large local-only artifacts generated during replay prep, trace collection, and training

## Tracked files

These stay in git because the code depends on them directly:
- `abilities.json`
- `items.json`
- `learnsets.json`
- `moves.json`
- `moveset_priors.json`
- `pokedex.json`
- `typechart.json`
- `ladder_accounts.example.txt`

## Local-only files

These are intentionally ignored by git:
- replay dumps such as `data/gen9random/`
- generated pickles such as `data/*.pkl`
- local ladder credentials in `data/ladder_accounts.txt`
- other machine-specific collection artifacts

## Ladder credentials

To use ladder scripts locally:

```bash
cp data/ladder_accounts.example.txt data/ladder_accounts.txt
```

Then fill in real credentials in `data/ladder_accounts.txt`.
