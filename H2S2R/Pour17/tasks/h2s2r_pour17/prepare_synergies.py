"""Fit and freeze the two five-dimensional H2S2R hand synergies.

The source corpus is the same estimated Sharpa retargeting contained in the
declared Pour17 input archive.  This is a preparation step, not a learned reward
or an additional task prior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rl_rebuild.baselines.h2s2r.pour17.inputs import load_inputs
from rl_rebuild.baselines.h2s2r.synergy import fit_pca_synergy, save_synergy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_root", type=Path, required=True)
    parser.add_argument("--input_archive", type=Path, default=None)
    parser.add_argument("--input_regime", choices=("estimated", "oracle"), default="estimated")
    parser.add_argument("--provenance_manifest", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve()
    perception_path = (
        bundle_root / "perception/pour17_perception.npz"
        if args.input_archive is None
        else args.input_archive.resolve()
    )
    inputs = load_inputs(
        perception_path,
        regime=args.input_regime,
        provenance_manifest=args.provenance_manifest,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, dict] = {}
    for side, samples, valid in (
        ("right", inputs.right_hand_q, inputs.right_valid),
        ("left", inputs.left_hand_q, inputs.left_valid),
    ):
        selected = samples[valid]
        synergy = fit_pca_synergy(
            selected,
            source=(
                f"pour17_{args.input_regime}_{side}_hand_q_valid_rows_pca5;"
                f"input_sha256={inputs.provenance.sha256}"
            ),
        )
        path = args.output_dir / f"{side}_synergy_pca5.npz"
        save_synergy(path, synergy)
        outputs[side] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "rows_used": int(valid.sum()),
            "source": synergy.source,
        }

    manifest = {
        "schema": "h2s2r_synergy_manifest_v1",
        "input_archive": (
            "perception/pour17_perception.npz"
            if perception_path == bundle_root / "perception/pour17_perception.npz"
            else str(perception_path)
        ),
        "input_regime": args.input_regime,
        "task_supervision_kind": inputs.provenance.task_supervision_kind.value,
        "input_sha256": inputs.provenance.sha256,
        "fit": {
            "method": "PCA/SVD",
            "components": 5,
            "lower_percentile": 0.5,
            "upper_percentile": 99.5,
            "valid_rows_only": True,
        },
        "outputs": {
            side: {**record, "path": Path(record["path"]).name}
            for side, record in outputs.items()
        },
    }
    manifest_path = args.output_dir / "synergy_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
