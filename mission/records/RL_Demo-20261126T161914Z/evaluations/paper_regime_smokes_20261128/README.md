# Paper human-regime Gazebo smoke runs

These artifacts record one-update pretrained PPO integration smokes on the two new nominal human-regime profiles, using only each profile's train split. The randomized run processed 20 transitions and the nonrandomized run processed one. Both loaded the official randomized GST actor and completed successfully through Gazebo Harmonic, DR-SPAAM, and PPO.

The randomized run started four episodes but none lasted to the five-second goal-change boundary, so its observed goal-change count is zero. Deterministic helper and PerWorld tests cover the exact boundary and hook. The nonrandomized run also recorded zero goal changes. These are wiring checks only; they do not establish policy quality or convergence. Neither run read the test split and neither actuated a physical robot.

The initial failed launch attempts and their causes are documented in smoke_report.json. The exact training summaries and environment counters are retained here; checkpoints remain in the external H100 run directory and are not bundled in Git.
