# H2S2R Clean3

Current handoff: [current_20260916](current_20260916/README.md).

The current experiment is an H2S2R comparison baseline with two disclosed
adaptations: direct 58D control (FABRICS/PCA are not used for the different
robot hands) and an Ours release-row Stage-1 grasp initializer before H2S2R
Stage-2. Do not confuse the Stage-1 367D/22D Ours architecture and reward with
the H2S2R Stage-2 342D/509D contract and task reward.

The older `deployment_c3p1r2`, `decisions`, `preparation`, `tests`, and
`evidence` directories are retained as the 2026-09-12/13 audit trail. Their
PCA/FABRICS/22-action launch path and blocked status are historical and must not
be used for new training.

Read in order:

1. [Current method boundary](current_20260916/METHOD_AND_DEVIATIONS.md)
2. [Deployment and runbook](current_20260916/DEPLOYMENT_AND_RUNBOOK.md)
3. [Active training evidence](current_20260916/TRAINING_STATUS_20260916.md)
