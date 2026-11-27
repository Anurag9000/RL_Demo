# One-case action and safety-gate diagnostic

- Checkpoint: seed 437 update 300, SHA-256 `0ac711418ff64269aef11ab3a55f92c9426a12dd3f5b1ed30bf34264f4d6cb37`.
- Validation scenario: `v5-validation-0085`; 1 episode, 9 policy transitions; outcome Collision.
- The gate blocked 34/45 physics substeps; 5/45 had a sensor/tracker fault, including 29 blocked substeps with no fault.
- At the first five decisions it tracked four humans and scans were 0.05-0.20 s old. Later, one five-substep decision had a detector/tracker fault and zero tracked humans.
- Thus both valid-perception gate blocks and a brief perception fault occurred. The code applies an all-around minimum-range cutoff and returns zero for both linear and angular commands. One case does not prove this explains all 500-case failures.
- Offline truth-trace verification passed. Diagnostic only; no checkpoint selected and the test split remains sealed.
