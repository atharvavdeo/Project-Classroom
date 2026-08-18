"""Product Zero pipeline (PRD.md, RTX 3090 handoff).

New infrastructure lives here: the immutable run folder, the run manifest,
configuration validation, preflight, and the five-state gate protocol.

Existing measured code in `classroom/` is *adapted*, not rewritten. Modules move
across one at a time, each behind a test, so that no working and measured
component is replaced by an unmeasured one.
"""
