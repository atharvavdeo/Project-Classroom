# Quality limitations — run `1446/01_phone`

Every limitation below is measured from this run's artifacts.

## Source resolution

7 of 1076 crops (0.7%) are below the native-resolution floor for an object call. These are abstentions by construction: no detector, slicing configuration or VLM can recover detail the source never captured.

Median magnification 3.73x, maximum 13.33x.

## Blocking inputs

- **Product-owner approval of the taxonomy.** PRD 3 blocks detector selection and evaluation until the target and hard-negative class list is approved.
- **rf_detr_native** — skipped by --no-rfdetr; it stays in the comparison as unavailable and is never substituted by D-FINE (PRD 5).
- **rf_detr_sahi** — skipped by --no-rfdetr; it stays in the comparison as unavailable and is never substituted by D-FINE (PRD 5).

## What cannot be claimed

- No accuracy, precision, recall or false-positives-per-hour figure, in the absence of a reference manifest.
- No statement about 30-60 person halls; the validated corpus is 1-17 people.
- No long-recording or events-per-hour claim; the corpus is short pre-cut clips processed end to end.
- No claim about any individual. The output is observational evidence for human review.
