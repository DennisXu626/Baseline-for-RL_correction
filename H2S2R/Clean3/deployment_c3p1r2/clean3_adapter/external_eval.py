"""Frozen Clean3 external evaluator v1, independent from the training reward."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Clean3ExternalEvaluatorV1:
    """Accumulate one 400-control episode without an early-success latch."""

    table_z: float = 0.87
    control_hz: float = 20.0
    required_controls: int = 400
    coverage_threshold: float = 0.45
    travel_threshold_m: float = 0.70
    rim_radius_m: float = 0.085
    grid_m: float = 0.01
    _covered: set[tuple[int, int]] = field(default_factory=set)
    _eligible_cells: set[tuple[int, int]] = field(default_factory=set)
    _controls: int = 0
    _travel_m: float = 0.0
    _previous_center: np.ndarray | None = None
    _previous_contact: bool = False
    _plate_initial: np.ndarray | None = None
    _max_plate_offset_m: float = 0.0
    _max_plate_tilt_deg: float = 0.0
    _far_pad_run: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.int64))
    _dropped: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=bool))
    _deep_run: int = 0
    _deep_penetration: bool = False

    def __post_init__(self) -> None:
        centers = np.arange(-0.085, 0.085 + 1e-12, self.grid_m)
        for x in centers:
            for y in centers:
                if x * x + y * y <= self.rim_radius_m**2:
                    self._eligible_cells.add(self._cell(np.array([x, y])))

    def _cell(self, xy: np.ndarray) -> tuple[int, int]:
        return tuple(np.floor((np.asarray(xy) + 0.09) / self.grid_m).astype(int))

    def update_physics(self, penetration_depth_m) -> None:
        """Consume independently computed hand-object/object-object penetration."""

        values = np.asarray(penetration_depth_m, dtype=float)
        deep = bool(values.size and np.isfinite(values).all() and values.max() > 0.002)
        self._deep_run = self._deep_run + 1 if deep else 0
        self._deep_penetration |= self._deep_run >= 2

    def update_control(
        self,
        *,
        plate_center,
        plate_normal,
        sponge_center_plate_xy,
        footprint_points_plate_xy,
        footprint_gap_m,
        plate_sponge_contact_force_n: float,
        object_centers_z,
        all_pad_surface_distances_m,
    ) -> None:
        plate_center = np.asarray(plate_center, dtype=float)
        plate_normal = np.asarray(plate_normal, dtype=float)
        sponge_center = np.asarray(sponge_center_plate_xy, dtype=float)
        footprint = np.asarray(footprint_points_plate_xy, dtype=float)
        gaps = np.asarray(footprint_gap_m, dtype=float)
        pad_dist = np.asarray(all_pad_surface_distances_m, dtype=float)
        if plate_center.shape != (3,) or plate_normal.shape != (3,):
            raise ValueError("plate center/normal must be 3-vectors")
        if footprint.ndim != 2 or footprint.shape[1] != 2 or gaps.shape != (len(footprint),):
            raise ValueError("footprint and gap arrays are inconsistent")
        if pad_dist.ndim != 2 or pad_dist.shape[0] != 2:
            raise ValueError("pad distances must have shape (2, number_of_pads)")

        if self._plate_initial is None:
            self._plate_initial = plate_center.copy()
        self._max_plate_offset_m = max(
            self._max_plate_offset_m,
            float(np.linalg.norm(plate_center - self._plate_initial)),
        )
        normal = plate_normal / np.linalg.norm(plate_normal)
        self._max_plate_tilt_deg = max(
            self._max_plate_tilt_deg,
            float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))),
        )

        support = (gaps >= -0.002) & (gaps < 0.005)
        physical_contact = float(plate_sponge_contact_force_n) > 0.0
        contact = bool(physical_contact and support.any())
        if contact:
            for point in footprint[support]:
                cell = self._cell(point)
                if cell in self._eligible_cells:
                    self._covered.add(cell)
        if self._previous_center is not None and self._previous_contact and contact:
            self._travel_m += float(np.linalg.norm(sponge_center - self._previous_center))
        self._previous_center = sponge_center.copy()
        self._previous_contact = contact

        below_table = np.asarray(object_centers_z, dtype=float) < self.table_z
        all_far = np.all(pad_dist > 0.10, axis=1)
        self._far_pad_run = np.where(all_far, self._far_pad_run + 1, 0)
        self._dropped |= below_table | (self._far_pad_run >= int(0.5 * self.control_hz))
        self._controls += 1

    def result(self) -> dict:
        coverage = len(self._covered) / len(self._eligible_cells)
        complete = self._controls == self.required_controls
        components = {
            "complete_400_controls": complete,
            "coverage": coverage >= self.coverage_threshold,
            "two_endpoint_contact_travel": self._travel_m >= self.travel_threshold_m,
            "plate_tilt_entire_episode": self._max_plate_tilt_deg < 15.0,
            "plate_offset_entire_episode": self._max_plate_offset_m < 0.03,
            "no_drop": not bool(self._dropped.any()),
            "no_sustained_deep_penetration": not self._deep_penetration,
        }
        return {
            "schema": "h2s2r_clean3_external_eval_v1",
            "controls": self._controls,
            "coverage": coverage,
            "two_endpoint_contact_travel_m": self._travel_m,
            "max_plate_tilt_deg": self._max_plate_tilt_deg,
            "max_plate_offset_m": self._max_plate_offset_m,
            "drop_by_object": self._dropped.tolist(),
            "sustained_deep_penetration": self._deep_penetration,
            "components": components,
            "success": bool(all(components.values())),
        }

