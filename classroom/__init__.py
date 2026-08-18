"""Project Classroom: offline examination-footage event prioritisation.

Pipeline phases, in strict execution order:

    1. ingest      - PyAV validation, PTS-accurate metadata
    2. calibration - seat polygons, zones, masks
    3. motion      - codec-MV whole-video scan
    4. segment     - per-seat hysteresis event boundaries
    5. api / ui    - investigator console
    6. detect      - D-FINE + RTMO on candidates
    7. verify      - Gemma verifier, GBNF-constrained

Phases 1-5 require no GPU.
"""

__version__ = "3.0.0"
