# Current Clean3 result summary

As of the 2026-09-16 snapshot, the exact Ours Stage-1 reproduction passed its
367D/22D/58D contract gate and PPO smoke, learned meaningful grasp behavior,
and annealed from release row 50 to row 40. At 3.817M/5M steps it had not yet
reached the strict row-10 Stage-2 gate.

Two deterministic single-episode three-view replays completed at 20 Hz: the
best checkpoint at row 50 and epoch-120 last checkpoint at row 45. Hashes and
metrics are in [the current status](current_20260916/TRAINING_STATUS_20260916.md).

No H2S2R Stage-2 result is claimed. The state-bank and residual-origin handoff
remain pending. Historical C3P1R2 failures remain in existing subdirectories
but are no longer the current execution route.
