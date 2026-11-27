# Mission decisions

## D-001 — Separate physical contacts from conservative terminal proxies

The current Gazebo evaluator's `Collision` category combines a Gazebo worker contact-or-tilt flag, continuous robot-human disk overlap, and wall-disk overlap. The seed436 update280 100-case validation run has negative minimum human-disk clearance in all 100 episodes but did not archive per-substep contact attribution. A matched one-episode probe found zero worker contact-or-tilt flags during the five scored policy decisions and a human-disk-overlap terminal at -0.008262 m; contact-or-tilt flags appeared in the unscored human-only suffix. Therefore, do not call the 100 outcomes verified physical contacts. Preserve the disk safety proxy and gate for now. Add per-episode outcome-source telemetry and run the matched 100-case validation probe before changing collision termination, reward or safety logic. Any changed semantics require a new training lineage.
