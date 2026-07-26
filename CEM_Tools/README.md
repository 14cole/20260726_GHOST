# CEM Tools

A standalone PySide6 application and headless Python/CLI library for GRIM
dataset library operations. The desktop interface is registry-driven: adding a
future tool means registering its callable and fields in
`cem_tools/registry.py`; the window builds its form automatically.

## Install and launch

```bash
cd CEM_Tools
python3 -m pip install -r requirements.txt
python3 -m cem_tools
```

By default conversion uses:

`/Users/emery/Documents/WORK/GRIM_Revised_2/grim_dataset.py`

On another workstation or HPC system, point to that project's folder:

```bash
export GRIM_REVISED_2_PATH=/path/to/GRIM_Revised_2
export CEM_SOLVER_BACKEND_PATH=/path/to/solver-project/Backend
```

## Headless/HPC use

All operations are ordinary Python functions:

```python
from cem_tools import subtract_datasets

result = subtract_datasets("/data/OPN", "/data/FRD", "/data/delta")
print(result.summary())
```

The CLI avoids importing PySide6:

```bash
python3 -m cem_tools.cli subtract OPN FRD Delta
python3 -m cem_tools.cli concat-pols Original Joined_Pols
python3 -m cem_tools.cli concat-freqs Original Joined_Freqs
python3 -m cem_tools.cli rename Original SEAL SEAM --output-dir Renamed
python3 -m cem_tools.cli rename Original old new --in-place
python3 -m cem_tools.cli convert Original Converted .grim
```

Add `--overwrite` only when existing outputs should be replaced.

## Dataset behavior

- Subtraction is `OPN - FRD` and is performed on float64
  `rcs_amp_real`/`rcs_amp_imag`, never on dB values or reconstructed display
  fields. It requires the current solver's explicit canonical 2D phase,
  amplitude, field-domain, and sigma/dBke metadata; legacy ambiguous files are
  rejected instead of being silently mislabeled as placement-ready deltas.
- OPN/FRD pairing uses the solver's canonical shared matcher. Studies must
  match, and every FRD parameter must occur with the same value in the OPN.
  The OPN may contain additional feature-only variables, allowing one clean
  FRD baseline to serve many featured cases. The most-specific compatible FRD
  wins; equally specific alternatives are rejected as ambiguous. Subtraction
  inputs require final `_OPN` and `_FRD` filename markers.
- Input files are matched from underscore-separated filename tokens.
  Polarizations recognized are VV, HH, VH, HV, TM and TE; frequency tokens
  such as `3GHz` or `0.0100GHz` are recognized wherever they appear.
- Concatenation verifies all non-joined axes and physical convention metadata,
  detects conflicting overlap, and keeps solver raw complex arrays.
- Conversion inputs are `.grim`, `.out`, `.pio`, `.cmplx_di`, `.csv`, `.txt`,
  and `.ss`. Outputs are `.grim`, `.pio`, `.cmplx_di`, `.csv`, `.txt`, and
  `.out`. `.ss` remains read-only because the referenced GRIM library has no
  supported writer.
- PIO and OUT formats only represent one elevation/polarization slice, so a
  multi-slice input produces explicitly suffixed output files.
- OUT output is allowed only for datasets tagged as 2D `sigma_2d`/`dBke`.
- `.grim` is the lossless choice for solver artifacts. CSV, TXT, PIO, and OUT
  do not have fields for all solver convention/provenance metadata and should
  be treated as exchange or plotting formats, not feature-placement inputs.

Operations inspect files directly in the selected folder and do not recurse.
Input and output folders must differ for subtraction, concatenation, and
conversion. Rename Files is the only tool that permits in-place changes. The
GUI reports batch summaries rather than listing every output file, so very
large libraries do not flood or retain a large results table.
