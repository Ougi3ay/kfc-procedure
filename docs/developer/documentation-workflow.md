# Documentation Workflow

## Install docs dependencies

```bash
python -m pip install mkdocs mkdocs-material pymdown-extensions "mkdocstrings[python]"
```

## Preview locally

```bash
python -m mkdocs serve -f mkdocs.yml
```

## Build strictly

```bash
python -m mkdocs build --strict
```

## Deploy manually

```bash
python -m mkdocs gh-deploy --force
```

## Recommended editing flow

1. Update Markdown files in `docs/`.
2. Preview locally.
3. Check code blocks, Mermaid diagrams, and LaTeX rendering.
4. Run `mkdocs build --strict`.
5. Commit docs with source changes.
