# Active Stage-1 evidence snapshot

Active run: `/ssd/sy/kailang/clean3/curriculum_20260915/gates/ours_stage1_5m_gpu1_20260915T1830Z`, seed 42, 512 environments, 5M target.

- Contract gate passed (367D actor, 22D privileged critic, 58D action).
- PPO smoke completed 65,536 steps; checkpoint SHA256 `94e4305d313cd16b154ea52dc7bc64d0b27ce6c1fae4c29fd4c726555ee93aab`.
- Actor unfroze after ten critic-only warmup epochs.
- Annealed `50 -> 45` near 1.77M and later `45 -> 40`.
- At 3,817,472 steps / epoch 232: row 40, success 53.42%, certification
  65.75%, drop 1.37%, EMA 55.83%, plate/sponge errors 1.16/1.25 cm.

This demonstrates learning and two transitions, but does not satisfy row 10.

Deterministic videos completed at 20 Hz: best at row 50, SHA256
`274f9af6b8bc4b73b0cfa86f9238872f50add6e731e2cd41cefbbec3c0a716e6`;
epoch-120 last at row 45, SHA256
`0b0de1fc57f4873ef3b94d5503259b457bd2acabc24b7562729402a422158267`.
Checkpoints/videos are not committed; they remain on the authorized server and
in the local experiment report.
