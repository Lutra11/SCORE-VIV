# Dataset area

Data sources remain independent and provenance-aware:

- `raw/`: immutable publisher/author files; normally restored by workflow cache.
- `digitized/`: values extracted from published figures with calibration notes.
- `processed/`: SI-normalized benchmark tables ready for algorithms.
- `generated/`: LF/MF/HF simulation data; not committed.
- `manifests/`: URLs, DOI, licenses, expected hashes, versions, and download status.
- `schemas/`: field and metadata contracts.

Xu 2023 VIV and Yu 2024 scour data are separate validation packages, not one
fully coupled public experiment.

