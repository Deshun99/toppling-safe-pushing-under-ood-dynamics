# SMC² milestone evidence — 24 October 2026

This bundle preserves the numerical evidence needed for the report and presentation.
It is stored outside ignored `artifacts/` and is eligible for Git tracking. The archive,
manifest, checksums, and this guide must be committed together to survive a fresh clone.
Creating these files alone does not commit or upload them.

## Primary results

Use these archived paths, in this order:

1. `artifacts/sysid_covariance_vs_cma_retained/report.md`: current production
   covariance SMC² versus retained CMA-ES under matched cumulative replay limits.
   Position RMSE is 0.111/0.141 mm for SMC² and 0.070/0.018 mm for CMA-ES
   (varied durations/passive mixed). Costs are 45,088 versus 45,085 fitting
   replays. SMC² has lower varied-duration COM error: 5.02 versus 8.74 mm.
2. `artifacts/sysid_covariance_proposals/report.md`: controlled sampler improvement.
   The capped covariance arm reduces mean COM grid distance from 11.05 to
   2.50 mm, preserves 212–229 distinct particles instead of 15–42, and uses
   about 14.4% fewer fitting replays. Its 0.082/0.126 mm prediction errors
   belong to the capped diagnostic, not the production comparison above.
3. `artifacts/sysid_truth_particle_trace/report.md`: oracle diagnostic explaining
   the difference between likelihood scoring, accepted moves, and ancestry loss.
   Do not include these privileged runs in performance comparisons.
4. `artifacts/sysid_reference_posterior/report.md`: separate numerical posterior
   calculation, its convergence evidence, and comparison with earlier SMC².

All other selected runs are supporting development history. The milestone is
based on one noise-free object, two main training sequences and three inference
seeds. It does not establish general calibration or better shield safety.

## Contents and deliberate exclusions

- Original reports, configurations, raw datasets, JSON/JSONL/CSV metrics and audits
  from 18 selected experiment directories.
- All particle snapshots in those directories, including their update histories.
- Both final independent grid calculations (`grid12_level5.npz` and
  `grid24_level5.npz`), with refinement/convergence logs.
- Summary figures, including COM distribution comparisons and the truth-survival plot.
- A closure source/test/dependency snapshot under `source/`, plus the handoff and
  canonical project outline. These are ordinary files for inspection, not an environment.
- `manifest.json` records every payload file's path, byte count and SHA-256,
  with an explicit list of omitted files from the selected directories.

Intermediate grid checkpoints and repeated per-update PNGs are omitted to reduce
size. Numerical particle snapshots remain available for redrawing the plots.
Unselected scratch/smoke/rerun directories, training checkpoints, the virtual
environment and literature PDFs are not included. No original artifact is moved
or deleted. This is a selected evidence archive, not a backup of the whole workspace.

## Integrity verification

Run from the repository root using the standard-library Python interpreter:

```bash
shasum -a 256 -c docs/findings/evidence/smc2_2026-10-24/SHA256SUMS
python3 - <<'PYCODE'
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
p = Path('docs/findings/evidence/smc2_2026-10-24')
m = json.loads((p / 'manifest.json').read_text())
with ZipFile(p / 'smc2_evidence.zip') as z:
    assert z.testzip() is None
    assert z.read('manifest.json') == (p / 'manifest.json').read_bytes()
    assert set(z.namelist()) == {f['path'] for f in m['files']} | {'manifest.json'}
    for f in m['files']:
        data = z.read(f['path'])
        assert len(data) == f['bytes'], f['path']
        assert hashlib.sha256(data).hexdigest() == f['sha256'], f['path']
print('Archive CRC and every payload checksum verified.')
PYCODE
```

## Restore the artifact links

In a fresh checkout, verify the archive first, then extract only its artifact entries:

```bash
unzip docs/findings/evidence/smc2_2026-10-24/smc2_evidence.zip 'artifacts/*' -d .
```

This restores the selected original paths used by the handoff at
`docs/findings/smc2_development_handoff.md`. Source snapshot entries are not copied
into the checkout by this command. Extract the complete archive to a separate
folder if you want to inspect the closure source snapshot and internal manifest.

## Provenance and reproduction limits

Runtime code and tests at closure correspond to commit
`741767376d7bc72951885f10b1af320361015bd0`. The source snapshot of the handoff
includes the new closure index added after that commit. The bundle's own commit
is necessarily later; inspect Git history for its identity.

Historical configs retain their original commit, dirty-tree flag, source hashes,
dependency versions and paths where originally recorded. Some older runs were
made with uncommitted code or options that were subsequently removed. The closure
snapshot is not a claim to contain the exact historical source of every run.
Preserved raw evidence and checksum integrity do not imply bitwise rerun results
on different environments. No Python environment or PyBullet binary is bundled.

Old reports may mention the former diagnostic module paths; current diagnostic
entry points live under `studies.diagnostics`. Covariance is now the estimator,
benchmark and demo default; historical fixed-proposal studies explicitly select
`fixed`. Preserve the distinction between the capped covariance diagnostic and
normal-schedule covariance SMC². Do not bypass source/data checks when attempting
to reuse old grid caches: absolute paths and source hashes may legitimately differ.
