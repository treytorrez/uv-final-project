# Executive Summary (Quarto)

This folder contains a Quarto website mock-up for a professional executive summary.

## Render

From repo root:

```bash
quarto render docs/executive_summary
```

Output: `docs/executive_summary/_site/index.html`

## Notes

- Theme: Bootswatch `darkly` + `styles.scss` overrides.
- The metrics table is currently populated with the latest baseline numbers you reported in the notebook; we can update it to auto-load from `models/meta_*.json` once your training/eval protocol is finalized.
