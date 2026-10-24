"""Prepare Exp 06 data: CIFAR-100 (target) and Tiny-ImageNet-200 (source) as uint8 .npy at 32x32 + sha256 manifest.

Outputs in --out (default $EXP06_ROOT/data):
  cifar100_{train,test}_{x,y}.npy      x: (N,32,32,3) uint8, y: (N,) int64
  tin200_{train,val}_{x,y}.npy         Tiny-ImageNet resized 64->32 with an exact 2x2 box filter
  manifest.json                         sha256 of every raw archive and every .npy, shapes, class lists

Usage:
  python prep_data.py --out $EXP06_ROOT/data --raw $EXP06_ROOT/raw          # download (needs internet) + convert
  python prep_data.py --out D --raw R --no-download                          # convert archives already in R
  python prep_data.py --out D --synthetic                                    # tiny learnable fake data for CPU smoke
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import pickle
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
CIFAR_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'  # torchvision's published checksum for this archive
TIN_URL = 'https://cs231n.stanford.edu/tiny-imagenet-200.zip'
COUNTS = {'cifar100_train': 50000, 'cifar100_test': 10000, 'tin200_train': 100000, 'tin200_val': 10000}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def download(url: str, dst: Path):
    if dst.exists() and dst.stat().st_size > 0:
        return
    tmp = dst.with_suffix(dst.suffix + '.part')
    print(f'downloading {url} -> {dst}', flush=True)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)


def cifar100(raw: Path, allow_download: bool):
    """CIFAR-100 via torchvision when installed (it downloads and md5-checks the same archive), else the archive."""
    try:
        import torchvision  # noqa: F401
        from torchvision.datasets import CIFAR100
        out = {}
        for split, train in (('train', True), ('test', False)):
            ds = CIFAR100(root=str(raw), train=train, download=True)  # with the archive present: md5 check + extract only
            out[split] = (np.asarray(ds.data, dtype=np.uint8), np.asarray(ds.targets, dtype=np.int64))
        classes = list(ds.classes)
        archive = raw / 'cifar-100-python.tar.gz'
        return out, classes, archive
    except ImportError:
        pass
    archive = raw / 'cifar-100-python.tar.gz'
    if allow_download:
        download(CIFAR_URL, archive)
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    assert md5 == CIFAR_MD5, f'CIFAR-100 archive md5 {md5} != {CIFAR_MD5}'
    out = {}
    with tarfile.open(archive) as tf:
        meta = pickle.load(tf.extractfile('cifar-100-python/meta'), encoding='latin1')
        for split in ('train', 'test'):
            d = pickle.load(tf.extractfile(f'cifar-100-python/{split}'), encoding='latin1')
            x = np.asarray(d['data'], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            out[split] = (np.ascontiguousarray(x), np.asarray(d['fine_labels'], dtype=np.int64))
    return out, list(meta['fine_label_names']), archive


def box_down(img_bytes: bytes):
    from PIL import Image
    im = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    a = np.asarray(im, dtype=np.float32)
    assert a.shape == (64, 64, 3), a.shape
    a = a.reshape(32, 2, 32, 2, 3).mean(axis=(1, 3))  # exact 2x2 box filter
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def tiny_imagenet(raw: Path, allow_download: bool):
    archive = raw / 'tiny-imagenet-200.zip'
    if allow_download:
        download(TIN_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        wnids = [s.strip() for s in zf.read('tiny-imagenet-200/wnids.txt').decode().split('\n') if s.strip()]
        assert len(wnids) == 200
        cls = {w: i for i, w in enumerate(sorted(wnids))}
        names = set(zf.namelist())
        tr = sorted(n for n in names if n.startswith('tiny-imagenet-200/train/') and n.endswith('.JPEG'))
        xs, ys = [], []
        for n in tr:
            xs.append(box_down(zf.read(n)))
            ys.append(cls[n.split('/')[2]])
        train = (np.stack(xs), np.asarray(ys, dtype=np.int64))
        ann = zf.read('tiny-imagenet-200/val/val_annotations.txt').decode().strip().split('\n')
        xs, ys = [], []
        for line in sorted(ann):
            f, w = line.split('\t')[:2]
            xs.append(box_down(zf.read(f'tiny-imagenet-200/val/images/{f}')))
            ys.append(cls[w])
        val = (np.stack(xs), np.asarray(ys, dtype=np.int64))
    return {'train': train, 'val': val}, sorted(wnids), archive


def synthetic(seed=0, n_src=(2000, 400), n_tgt=(2000, 400)):
    """Learnable fake data: each class is a fixed random 32x32x3 template plus noise; source/target share templates
    in part so transfer is possible. Only for the CPU smoke; never for measurement."""
    rng = np.random.default_rng(seed)
    # 10 classes per domain (labels stay valid for the 200/100-way heads); coarse 8x8-block templates so a few
    # hundred CPU steps visibly lower the loss.
    base = np.kron(rng.integers(0, 256, size=(20, 4, 4, 3)), np.ones((1, 8, 8, 1))).astype(np.float32)

    def make(n, offset):
        y = rng.integers(0, 10, size=n)
        x = 0.8 * base[y + offset] + 0.2 * rng.integers(0, 256, size=(n, 32, 32, 3))
        return np.clip(x, 0, 255).astype(np.uint8), y.astype(np.int64)

    return ({'train': make(n_src[0], 0), 'val': make(n_src[1], 0)},
            {'train': make(n_tgt[0], 5), 'test': make(n_tgt[1], 5)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--raw', default=None, help='where the archives live / are downloaded (default <out>/../raw)')
    ap.add_argument('--no-download', action='store_true')
    ap.add_argument('--synthetic', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    manifest = {'created_unix': int(t0), 'synthetic': a.synthetic, 'files': {}, 'raw': {}}
    if a.synthetic:
        tin, cif = synthetic()
        tin_classes, cif_classes = [f's{i}' for i in range(200)], [f't{i}' for i in range(100)]
    else:
        raw = Path(a.raw) if a.raw else out.parent / 'raw'
        raw.mkdir(parents=True, exist_ok=True)
        cif, cif_classes, cif_arch = cifar100(raw, not a.no_download)
        tin, tin_classes, tin_arch = tiny_imagenet(raw, not a.no_download)
        for p in (cif_arch, tin_arch):
            manifest['raw'][p.name] = {'sha256': sha256(p), 'bytes': p.stat().st_size}
    arrays = {'cifar100_train': cif['train'], 'cifar100_test': cif['test'],
              'tin200_train': tin['train'], 'tin200_val': tin['val']}
    for name, (x, y) in arrays.items():
        assert x.dtype == np.uint8 and x.shape[1:] == (32, 32, 3), (name, x.shape)
        assert len(x) == len(y)
        if not a.synthetic:
            assert len(x) == COUNTS[name], (name, len(x))
        for suf, arr in (('x', x), ('y', y)):
            p = out / f'{name}_{suf}.npy'
            np.save(p, arr)
            manifest['files'][p.name] = {'sha256': sha256(p), 'shape': list(arr.shape), 'dtype': str(arr.dtype)}
        manifest['files'][f'{name}_y.npy']['n_classes_present'] = int(len(np.unique(y)))
    manifest['classes'] = {'cifar100': cif_classes, 'tin200': tin_classes}
    manifest['seconds'] = round(time.time() - t0, 1)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    for k, v in manifest['raw'].items():
        print(f'raw {k} sha256={v["sha256"]}')
    for k, v in manifest['files'].items():
        print(f'{k} {v["shape"]} sha256={v["sha256"][:16]}')
    print(f'prep done in {manifest["seconds"]} s -> {out}/manifest.json')


if __name__ == '__main__':
    main()
