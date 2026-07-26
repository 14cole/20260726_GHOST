"""Lossless operations on solver-compatible GRIM/NPZ datasets.

This module deliberately avoids the viewer's float32 power/phase reconstruction
for coherent work. Solver ``rcs_amp_real``/``rcs_amp_imag`` arrays are the
authoritative fields and remain float64 through joins and subtraction.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Literal

import numpy as np

from .errors import CemToolError


C0 = 299_792_458.0
AXIS_KEYS = ("azimuths", "elevations", "frequencies", "polarizations")
REQUIRED_KEYS = AXIS_KEYS + ("rcs_power", "rcs_phase")
CRITICAL_METADATA = (
    "rcs_domain",
    "power_domain",
    "phase_reference",
    "amplitude_convention",
    "complex_field_domain",
    "units",
)
DELTA_FIELD_DOMAIN = "featured_minus_clean_far_field_amplitude_delta"
DELTA_PHASE_SUFFIX = (
    "; coherent subtraction=featured-clean; placement phase center is the "
    "seam line on the coupon outer face y=0"
)
PHYSICAL_2D_PHASE_REFERENCE = (
    "origin=(0,0), convention=exp(+jwt); stored complex field is the "
    "2D layer-potential bare-integral amplitude B. The coefficient "
    "in u_s~exp(-j(kr-pi/4))/sqrt(8*pi*k*r)*A is A=j*B."
)
PHYSICAL_2D_AMPLITUDE_CONVENTION = "A_physical_asymptotic = +j * B_stored"
PHYSICAL_2D_FIELD_DOMAIN = "2d_layer_potential_bare_integral_amplitude_B"


def _scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise CemToolError("expected scalar GRIM metadata")
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def _grid_shape(payload: dict[str, np.ndarray]) -> tuple[int, int, int, int]:
    return tuple(len(np.asarray(payload[key]).ravel()) for key in AXIS_KEYS)


def load_grim(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".grim" or not source.is_file():
        raise CemToolError(f"not a readable .grim file: {source}")
    try:
        with np.load(source, allow_pickle=False) as data:
            payload = {key: np.array(data[key], copy=True) for key in data.files}
    except (OSError, ValueError, TypeError) as exc:
        raise CemToolError(f"cannot read {source.name}: {exc}") from exc
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise CemToolError(
            f"{source.name} is missing required GRIM fields {missing}"
        )
    for key in AXIS_KEYS:
        axis = np.asarray(payload[key]).ravel()
        if axis.ndim != 1 or axis.size == 0:
            raise CemToolError(f"{source.name}: axis {key} is empty")
        payload[key] = axis
    shape = _grid_shape(payload)
    for key in ("rcs_power", "rcs_phase"):
        if np.asarray(payload[key]).shape != shape:
            raise CemToolError(
                f"{source.name}: {key} shape {np.shape(payload[key])} != {shape}"
            )
    for key in ("rcs_amp_real", "rcs_amp_imag"):
        if key in payload and np.asarray(payload[key]).shape != shape:
            raise CemToolError(
                f"{source.name}: {key} shape {np.shape(payload[key])} != {shape}"
            )
    return payload


def save_grim_atomic(
    payload: dict[str, Any],
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".grim":
        destination = destination.with_suffix(".grim")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise CemToolError(f"output exists: {destination}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **payload)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return destination


def raw_amplitude(payload: dict[str, np.ndarray], label: str) -> np.ndarray:
    if "rcs_amp_real" not in payload or "rcs_amp_imag" not in payload:
        raise CemToolError(
            f"{label}: coherent operations require preserved "
            "rcs_amp_real/rcs_amp_imag arrays"
        )
    amplitude = (
        np.asarray(payload["rcs_amp_real"], dtype=np.float64)
        + 1j * np.asarray(payload["rcs_amp_imag"], dtype=np.float64)
    )
    if not np.all(np.isfinite(amplitude)):
        raise CemToolError(f"{label}: raw complex amplitude is nonfinite")
    return amplitude


def _axis_equal(left: np.ndarray, right: np.ndarray, categorical: bool) -> bool:
    if categorical:
        return np.array_equal(
            np.asarray(left).astype(str), np.asarray(right).astype(str)
        )
    return np.array_equal(np.asarray(left, float), np.asarray(right, float))


def _critical_metadata_equal(
    payloads: list[dict[str, np.ndarray]], labels: list[str]
) -> None:
    for key in CRITICAL_METADATA:
        present = [key in payload for payload in payloads]
        if any(present) and not all(present):
            raise CemToolError(
                f"{key} metadata is missing from part of the group: {labels}"
            )
        if all(present):
            values = [_scalar_text(payload[key]) for payload in payloads]
            if len(set(values)) != 1:
                raise CemToolError(
                    f"incompatible {key} metadata in {labels}: {values}"
                )


def _union_axis(arrays: Iterable[np.ndarray], categorical: bool) -> np.ndarray:
    if categorical:
        values: list[str] = []
        for array in arrays:
            for raw in np.asarray(array).ravel():
                value = str(raw)
                if value not in values:
                    values.append(value)
        return np.asarray(values, dtype=str)
    values = sorted(
        {float(raw) for array in arrays for raw in np.asarray(array).ravel()}
    )
    return np.asarray(values, dtype=float)


def _indices(union: np.ndarray, values: np.ndarray, categorical: bool) -> list[int]:
    result: list[int] = []
    for raw in np.asarray(values).ravel():
        if categorical:
            matches = np.flatnonzero(union.astype(str) == str(raw))
        else:
            matches = np.flatnonzero(
                np.isclose(union.astype(float), float(raw), rtol=0.0, atol=1e-9)
            )
        if len(matches) != 1:
            raise CemToolError(f"cannot uniquely align axis value {raw!r}")
        result.append(int(matches[0]))
    return result


def _merge_metadata(
    payloads: list[dict[str, np.ndarray]],
    shape: tuple[int, int, int, int],
    history: str,
) -> dict[str, Any]:
    first = payloads[0]
    merged: dict[str, Any] = {}
    excluded = set(AXIS_KEYS) | {
        "rcs_power",
        "rcs_phase",
        "rcs_amp_real",
        "rcs_amp_imag",
        "source_path",
        "history",
        "solver_metadata_json",
        "polarization_alias_primary",
        "polarization_aliases_json",
    }
    for key, value in first.items():
        if key in excluded:
            continue
        array = np.asarray(value)
        if array.shape == _grid_shape(first):
            continue
        candidates = [payload.get(key) for payload in payloads]
        if all(candidate is not None for candidate in candidates):
            try:
                if all(
                    np.array_equal(np.asarray(candidates[0]), np.asarray(candidate))
                    for candidate in candidates[1:]
                ):
                    merged[key] = value
            except (TypeError, ValueError):
                pass
    merged["source_path"] = np.asarray("")
    merged["history"] = np.asarray(history)
    return merged


def join_payloads(
    payloads: list[dict[str, np.ndarray]],
    *,
    axis: Literal["polarizations", "frequencies"],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    if not payloads:
        raise CemToolError("no GRIM datasets to concatenate")
    labels = labels or [f"input {index}" for index in range(len(payloads))]
    if len(labels) != len(payloads):
        raise CemToolError("join labels do not match input count")
    for payload, label in zip(payloads, labels):
        has_real = "rcs_amp_real" in payload
        has_imag = "rcs_amp_imag" in payload
        if has_real != has_imag:
            raise CemToolError(
                f"{label}: raw complex amplitude must contain both real and imaginary arrays"
            )
    raw_presence = ["rcs_amp_real" in payload for payload in payloads]
    if any(raw_presence) and not all(raw_presence):
        raise CemToolError(
            "cannot concatenate a mixture of raw solver fields and "
            "magnitude/phase-only datasets"
        )
    _critical_metadata_equal(payloads, labels)
    join_index = AXIS_KEYS.index(axis)
    for key_index, key in enumerate(AXIS_KEYS):
        if key_index == join_index:
            continue
        categorical = key == "polarizations"
        reference = payloads[0][key]
        for payload, label in zip(payloads[1:], labels[1:]):
            if not _axis_equal(reference, payload[key], categorical):
                raise CemToolError(
                    f"{label}: {key} differs from the other files in its group"
                )
    axes = {
        key: (
            _union_axis(
                [payload[key] for payload in payloads],
                key == "polarizations",
            )
            if key == axis
            else np.array(payloads[0][key], copy=True)
        )
        for key in AXIS_KEYS
    }
    shape = tuple(len(axes[key]) for key in AXIS_KEYS)
    common_grid_keys = {
        key
        for key in payloads[0]
        if np.asarray(payloads[0][key]).shape == _grid_shape(payloads[0])
        and all(
            key in payload
            and np.asarray(payload[key]).shape == _grid_shape(payload)
            for payload in payloads
        )
    }
    common_grid_keys.update({"rcs_power", "rcs_phase"})
    merged_arrays: dict[str, np.ndarray] = {}
    filled: dict[str, np.ndarray] = {}
    for key in common_grid_keys:
        dtype = np.result_type(*[np.asarray(payload[key]).dtype for payload in payloads])
        if np.issubdtype(dtype, np.floating):
            array = np.full(shape, np.nan, dtype=dtype)
        elif np.issubdtype(dtype, np.complexfloating):
            array = np.full(shape, np.nan + 1j * np.nan, dtype=dtype)
        else:
            continue
        merged_arrays[key] = array
        filled[key] = np.zeros(shape, dtype=bool)
    for payload, label in zip(payloads, labels):
        selections = [
            _indices(
                axes[key],
                payload[key],
                key == "polarizations",
            )
            for key in AXIS_KEYS
        ]
        destination = np.ix_(*selections)
        for key, output in merged_arrays.items():
            incoming = np.asarray(payload[key])
            occupied = filled[key][destination]
            if np.any(occupied):
                existing = output[destination]
                if not np.allclose(
                    existing[occupied],
                    incoming[occupied],
                    rtol=1e-12,
                    atol=0.0,
                    equal_nan=True,
                ):
                    raise CemToolError(
                        f"{label}: overlapping {key} samples disagree"
                    )
            output[destination] = incoming
            filled[key][destination] = True
    for key, mask in filled.items():
        if not np.all(mask):
            raise CemToolError(f"concatenation left missing {key} cells")
    history = (
        f"CEM Tools concatenated {axis}: "
        + ", ".join(Path(label).name for label in labels)
    )
    result = _merge_metadata(payloads, shape, history)
    result.update(axes)
    result.update(merged_arrays)
    if "rcs_amp_real" in result and "rcs_amp_imag" in result:
        result["rcs_amp_real"] = np.asarray(result["rcs_amp_real"], np.float64)
        result["rcs_amp_imag"] = np.asarray(result["rcs_amp_imag"], np.float64)
        result["raw_complex_amplitude_preserved"] = np.asarray(True)
    return result


def _normalization_per_frequency(
    payloads: list[dict[str, np.ndarray]],
    amplitudes: list[np.ndarray],
    labels: list[str],
) -> np.ndarray:
    frequencies = np.asarray(payloads[0]["frequencies"], dtype=float)
    factors = np.empty(len(frequencies), dtype=float)
    for frequency_index, frequency in enumerate(frequencies):
        ratios: list[np.ndarray] = []
        for payload, amplitude in zip(payloads, amplitudes):
            power = np.asarray(payload["rcs_power"], dtype=float)[
                :, :, frequency_index, :
            ]
            magnitude_sq = np.abs(amplitude[:, :, frequency_index, :]) ** 2
            valid = (
                np.isfinite(power)
                & np.isfinite(magnitude_sq)
                & (magnitude_sq > np.finfo(float).tiny)
            )
            if np.any(valid):
                ratios.append(power[valid] / magnitude_sq[valid])
        if not ratios:
            raise CemToolError(
                f"cannot infer field normalization at {frequency:g} GHz"
            )
        values = np.concatenate(ratios)
        factor = float(np.median(values))
        if (
            not math.isfinite(factor)
            or factor <= 0.0
            or not np.allclose(values, factor, rtol=2e-5, atol=0.0)
        ):
            raise CemToolError(
                f"inconsistent power/amplitude normalization at "
                f"{frequency:g} GHz in {labels}"
            )
        factors[frequency_index] = factor
    return factors


def _validate_2d_source(payload: dict[str, np.ndarray], label: str) -> None:
    expected = {
        "rcs_domain": "power_phase",
        "power_domain": "linear_rcs",
        "phase_reference": PHYSICAL_2D_PHASE_REFERENCE,
        "amplitude_convention": PHYSICAL_2D_AMPLITUDE_CONVENTION,
        "complex_field_domain": PHYSICAL_2D_FIELD_DOMAIN,
    }
    for key, wanted in expected.items():
        if key not in payload:
            raise CemToolError(
                f"{label}: missing required 2D solver metadata {key!r}; "
                "legacy/ambiguous coherent fields cannot be subtracted safely"
            )
        actual = _scalar_text(payload[key])
        if actual != wanted:
            raise CemToolError(
                f"{label}: {key}={actual!r}; expected canonical 2D value {wanted!r}"
            )
    if not np.array_equal(
        np.asarray(payload["elevations"], dtype=float), np.asarray([0.0])
    ):
        raise CemToolError(f"{label}: a 2D source requires elevation axis [0.0]")
    try:
        units = json.loads(_scalar_text(payload["units"]))
    except (KeyError, json.JSONDecodeError, TypeError) as exc:
        raise CemToolError(f"{label}: invalid or missing units metadata") from exc
    if (
        units.get("rcs_linear_quantity") != "sigma_2d"
        or units.get("rcs_log_unit") != "dBke"
    ):
        raise CemToolError(
            f"{label}: coherent feature subtraction requires sigma_2d/dBke units"
        )


def subtract_payloads(
    featured: dict[str, np.ndarray],
    clean: dict[str, np.ndarray],
    *,
    featured_label: str,
    clean_label: str,
) -> dict[str, Any]:
    _validate_2d_source(featured, featured_label)
    _validate_2d_source(clean, clean_label)
    _critical_metadata_equal(
        [featured, clean], [featured_label, clean_label]
    )
    for key in AXIS_KEYS:
        if not _axis_equal(
            featured[key], clean[key], key == "polarizations"
        ):
            raise CemToolError(
                f"featured and clean {key} axes differ for "
                f"{featured_label} / {clean_label}"
            )
    featured_amp = raw_amplitude(featured, featured_label)
    clean_amp = raw_amplitude(clean, clean_label)
    factors = _normalization_per_frequency(
        [featured, clean],
        [featured_amp, clean_amp],
        [featured_label, clean_label],
    )
    delta = featured_amp - clean_amp
    power = (
        np.abs(delta) ** 2
        * factors.reshape(1, 1, -1, 1)
    )
    result = _merge_metadata(
        [featured, clean],
        delta.shape,
        f"CEM Tools coherent subtraction: {featured_label} - {clean_label}",
    )
    for key in AXIS_KEYS:
        result[key] = np.array(featured[key], copy=True)
    result.update(
        rcs_power=power.astype(np.float32),
        rcs_phase=np.angle(delta).astype(np.float32),
        rcs_amp_real=delta.real.astype(np.float64),
        rcs_amp_imag=delta.imag.astype(np.float64),
        raw_complex_amplitude_preserved=np.asarray(True),
        rcs_domain=np.asarray("delta"),
        power_domain=np.asarray("linear_rcs"),
        complex_field_domain=np.asarray(DELTA_FIELD_DOMAIN),
    )
    phase_reference = _scalar_text(featured.get("phase_reference", ""))
    if phase_reference:
        result["phase_reference"] = np.asarray(
            phase_reference + DELTA_PHASE_SUFFIX
        )
    return result
