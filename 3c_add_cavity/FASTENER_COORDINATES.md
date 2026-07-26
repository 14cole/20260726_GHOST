# Fastener coordinate libraries in step 3c

Step 3c can reuse one solved 3D differential pattern at every coordinate of the
same fastener type. It coherently accumulates all configured placements into
one component, `Output/<NAME>.grim`, for step 4.

## 1. Prepare one CSV per fastener type

Coordinates and vectors are in the CAD frame and the `UNITS` selected in
`run.py`:

- `+y`: nose
- `+x`: vehicle right
- `+z`: up

The recommended six-column form is:

```csv
x,y,z,nx,ny,nz
0.030,0.020,0.000,1.0,0.0,0.0
0.040,0.020,0.000,1.0,0.0,0.0
```

Accepted layouts are:

- `x,y,z`: derive the outward normal from the BoR body profile.
- `x,y,z,nx,ny,nz`: use and validate the supplied outward normal.
- `x,y,z,nx,ny,nz,rx,ry,rz`: also provide per-fastener clocking.

Headers are optional. Header aliases such as `x_normal`, `normal_x`, `x roll`,
and `roll_x` are accepted. Blank lines and lines beginning with `#` are ignored.

The normal is unitless and need not already be normalized. A supplied normal is
compared with the outward BoR skin normal and refused when it differs by more
than `NORMAL_TOL_DEG`.

The optional roll-reference vector fixes the local pattern frame's azimuth-zero
direction. It must not be parallel to the normal. For rotationally symmetric
fasteners its physical effect should vanish, but it still defines the numerical
frame.

## 2. Configure the pattern-to-CSV catalog

Edit the knobs in `run.py`:

```python
NAME = "fasteners"

FASTENER_TYPES = [
    {
        "name": "flush_rivet",
        "pattern": os.path.join("Patterns", "flush_rivet.grim"),
        "coordinates": os.path.join("Fasteners", "flush_rivet.csv"),
        "roll_ref": (0.0, 1.0, 0.0),
    },
    {
        "name": "bolt_head",
        "pattern": os.path.join("Patterns", "bolt_head.grim"),
        "coordinates": os.path.join("Fasteners", "bolt_head.csv"),
        "roll_ref": (0.0, 1.0, 0.0),
    },
]
```

The `roll_ref` in the catalog is used for rows that do not carry `rx,ry,rz`.
Leave `FASTENER_TYPES = []` to retain the original single-cavity mode.

## 3. Run

```bash
cd 3c_add_cavity
python3 run.py
```

The runner:

1. verifies the body artifact;
2. reads every coordinate table;
3. converts CAD coordinates and vectors to the solver frame;
4. checks each coordinate against the body skin and phase-error tolerance;
5. rejects reversed/inconsistent normals and duplicate coordinates;
6. verifies each unique compact-feature pattern;
7. applies orientation and two-way placement phase independently;
8. coherently sums all placements into `Output/<NAME>.grim`; and
9. commits pattern and CSV hashes, placement counts, validation summaries, and
   configuration to the provenance manifest. Coordinates are not duplicated
   into the manifest, so very large placement libraries remain compact.

Step 4 then discovers this one aggregate component through the normal component
manifest.

## Pattern requirement

Each pattern must be the complex 3D differential response:

```text
installed fastener on local skin - the same clean local skin
```

It must meet the compact-pattern convention enforced by
`Backend.feature_sum.point_pattern_convention_metadata()`. A free-space
fastener pattern is not an equivalent substitute because it omits interaction
with the supporting skin.
