"""Run from any directory: python3 task3/reconstruct.py. Requires numpy, scipy, Pillow."""
from pathlib import Path
from itertools import permutations
import json
import numpy as np
from PIL import Image
from scipy.ndimage import correlate

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'


def apply_filters(image, filters, mode='constant'):
    result = np.asarray(image, dtype=np.float64)
    for kernel in filters:
        result = correlate(result, kernel, mode=mode, cval=0.0)
    return result


def main():
    known = np.loadtxt(DATA / 'algos.csv', delimiter=',').reshape(2, 3, 3)
    a, b = known
    paths = sorted(DATA.glob('*.png'), key=lambda p: int(p.stem))
    # Only five pairs are used to estimate the nine unknown coefficients.
    fit_pairs = [(np.asarray(Image.open(p), dtype=float), np.loadtxt(p.with_suffix('.txt')))
                 for p in paths[:5]]
    candidates = []
    for mode in ['constant', 'reflect', 'mirror', 'nearest', 'wrap']:
        for order in permutations('ABX'):
            matrices, targets = [], []
            for image, target in fit_pairs:
                columns = []
                for basis in np.eye(9).reshape(9, 3, 3):
                    kernels = {'A': a, 'B': b, 'X': basis}
                    columns.append(apply_filters(image, [kernels[n] for n in order], mode).ravel())
                matrices.append(np.stack(columns, axis=1))
                targets.append(target.ravel())
            design, target = np.concatenate(matrices), np.concatenate(targets)
            coeffs, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
            mse = float(np.mean((design @ coeffs - target) ** 2))
            candidates.append(dict(order=''.join(order), mode=mode, mse=mse,
                                   rank=int(rank), kernel=coeffs.reshape(3, 3).tolist()))
    candidates.sort(key=lambda c: c['mse'])
    best = candidates[0]
    assert best['rank'] == 9 and best['mse'] < 1e-15
    # Recover exact short decimal coefficients; validate this rounding on every pair.
    missing = np.round(np.array(best['kernel']), 12)
    kernels = {'A': a, 'B': b, 'X': missing}
    validation = {}
    for order in ['XBA', 'BXA']:
        squared_error = 0.0
        pixel_count = 0
        max_error = 0.0
        for path in paths:
            target = np.loadtxt(path.with_suffix('.txt'))
            result = apply_filters(np.asarray(Image.open(path)), [kernels[n] for n in order])
            assert result.shape == target.shape
            error = result - target
            squared_error += float(np.sum(error ** 2))
            pixel_count += error.size
            max_error = max(max_error, float(np.max(np.abs(error))))
        validation[order] = dict(images=len(paths), pixels=pixel_count,
                                 mse=squared_error / pixel_count, max_abs_error=max_error)
        assert max_error == 0.0, validation[order]
    np.savetxt(ROOT / 'reconstructed_algos.csv', np.stack([missing, b, a]).reshape(3, 9),
               delimiter=',', fmt='%.12g')
    np.savetxt(ROOT / 'reconstructed_algos_alternative.csv', np.stack([b, missing, a]).reshape(3, 9),
               delimiter=',', fmt='%.12g')
    report = dict(missing_filter=missing.tolist(), boundary='zero padding, one pixel at each stage',
                  validation=validation, candidates=candidates,
                  note='XBA and BXA are observationally indistinguishable; original first two positions are not identifiable.')
    (ROOT / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['missing_filter', 'validation', 'note']}, indent=2))


if __name__ == '__main__':
    main()
