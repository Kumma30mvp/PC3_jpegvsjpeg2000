"""Real JPEG2000 (wavelet) codec with multiple backends.

Backend preference order:
    1. glymur          - directly accepts ``cratios`` so the cr5/cr10/cr20 targets
                         are honoured by the encoder.
    2. imagecodecs     - exposes a ``level`` quality knob; we map our cr targets
                         to quality levels and warn that the actual CR is only
                         enforced after we measure the file.
    3. Pillow JPEG2000 - uses ``quality_mode='rates'`` with a layer rate.
    4. OpenCV          - only attempted if the build supports JP2 I/O.

Whichever backend succeeds, the actual compression ratio reported in the CSV is
always computed from the real on-disk file size.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import config
from .io_utils import save_png

# JP2 file signature box (first 12 bytes of a real .jp2 file).
_JP2_SIGNATURE = bytes(
    [0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A]
)
# Raw J2K codestream marker (start-of-codestream).
_J2K_SOC = bytes([0xFF, 0x4F, 0xFF, 0x51])


def _is_real_jp2_file(path: Path) -> bool:
    """Confirm that ``path`` starts with the JP2 signature box or J2K SOC."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return False
    if len(head) >= 12 and head[:12] == _JP2_SIGNATURE:
        return True
    if len(head) >= 4 and head[:4] == _J2K_SOC:
        return True
    return False


# ---------------------------------------------------------------------------
# Per-backend encode/decode pairs.
# Each encoder writes the compressed bytes to ``out_path`` and returns nothing.
# Each decoder reads ``in_path`` and returns an HxWx3 uint8 RGB array.
# ---------------------------------------------------------------------------


def _encode_glymur(arr: np.ndarray, out_path: Path, cr: int) -> None:
    import glymur  # type: ignore
    if out_path.exists():
        out_path.unlink()
    glymur.Jp2k(str(out_path), data=arr, cratios=[int(cr)])


def _decode_glymur(in_path: Path) -> np.ndarray:
    import glymur  # type: ignore
    arr = np.asarray(glymur.Jp2k(str(in_path))[:])
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr.astype(np.uint8)


# Empirical mapping from "target compression ratio" to imagecodecs' quality level.
# imagecodecs does not accept cratios directly; the actual CR is recomputed from
# the resulting file size and reported in the CSV.
_IMAGECODECS_LEVEL_MAP = {5: 90, 10: 60, 20: 30}


def _encode_imagecodecs(arr: np.ndarray, out_path: Path, cr: int) -> None:
    import imagecodecs  # type: ignore
    level = _IMAGECODECS_LEVEL_MAP.get(int(cr))
    if level is None:
        warnings.warn(
            f"[jpeg2000][imagecodecs] no level mapping for cr={cr}; falling back to level=50",
            stacklevel=2,
        )
        level = 50
    encoded = imagecodecs.jpeg2k_encode(arr, level=int(level), codecformat="jp2")
    out_path.write_bytes(bytes(encoded))


def _decode_imagecodecs(in_path: Path) -> np.ndarray:
    import imagecodecs  # type: ignore
    data = in_path.read_bytes()
    arr = imagecodecs.jpeg2k_decode(data)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr.astype(np.uint8)


def _encode_pillow(arr: np.ndarray, out_path: Path, cr: int) -> None:
    from PIL import Image
    Image.fromarray(arr, mode="RGB").save(
        str(out_path),
        format="JPEG2000",
        quality_mode="rates",
        quality_layers=[float(cr)],
        irreversible=True,
    )


def _decode_pillow(in_path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(str(in_path)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _encode_opencv(arr: np.ndarray, out_path: Path, cr: int) -> None:
    import cv2
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    # OpenCV accepts a quality value 0..1000; lower means more compression.
    quality = max(1, min(1000, int(round(1000.0 / max(int(cr), 1)))))
    ok = cv2.imwrite(
        str(out_path),
        bgr,
        [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), int(quality)],
    )
    if not ok:
        raise RuntimeError("cv2.imwrite returned False for JPEG2000")


def _decode_opencv(in_path: Path) -> np.ndarray:
    import cv2
    bgr = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise RuntimeError(f"cv2.imread returned None for {in_path}")
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)


_BACKENDS: List[Tuple[str, Callable, Callable]] = [
    ("glymur-JPEG2000", _encode_glymur, _decode_glymur),
    ("imagecodecs-JPEG2000", _encode_imagecodecs, _decode_imagecodecs),
    ("Pillow-JPEG2000", _encode_pillow, _decode_pillow),
    ("OpenCV-JPEG2000", _encode_opencv, _decode_opencv),
]


def _backend_round_trip(
    encode: Callable, decode: Callable, tmp_dir: Path, cr: int = 10
) -> bool:
    """Round-trip a deterministic test image and verify shape, dtype, and JP2 magic."""
    rng = np.random.default_rng(42)
    test = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    out_path = tmp_dir / f"_jp2_check_{int(time.time()*1e6)}.jp2"
    try:
        encode(test, out_path, cr)
        if not out_path.exists() or out_path.stat().st_size <= 0:
            return False
        if not _is_real_jp2_file(out_path):
            return False
        recon = decode(out_path)
        if recon.shape != test.shape or recon.dtype != np.uint8:
            return False
        return True
    except Exception:
        return False
    finally:
        try:
            if out_path.exists():
                out_path.unlink()
        except OSError:
            pass


_active_backend: Optional[Tuple[str, Callable, Callable]] = None


def check_jpeg2000_backend() -> str:
    """Return the name of the first backend that survives a strict round-trip.

    Caches the choice so subsequent calls are free.
    Raises ``RuntimeError`` with the exact spec message if every backend fails.
    """
    global _active_backend

    if _active_backend is not None:
        return _active_backend[0]

    tmp_dir = config.COMPRESSED_DIR / "jpeg2000"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    for name, enc, dec in _BACKENDS:
        try:
            if _backend_round_trip(enc, dec, tmp_dir):
                _active_backend = (name, enc, dec)
                return name
            failures.append(f"{name}: round-trip check failed")
        except ImportError as e:
            failures.append(f"{name}: import error ({e})")
        except Exception as e:  # noqa: BLE001 - try every backend
            failures.append(f"{name}: {type(e).__name__}: {e}")

    detail = "\n  - " + "\n  - ".join(failures) if failures else ""
    raise RuntimeError(
        "No functional JPEG2000 backend was found. Install imagecodecs or "
        "configure OpenJPEG/glymur. The JPEG experiment can run, but JPEG2000 "
        "cannot." + detail
    )


def encode_decode_jpeg2000(
    image_uint8: np.ndarray,
    cr: int,
    image_stem: str,
    level_name: str,
    backend_name: Optional[str] = None,
) -> Dict:
    """Encode ``image_uint8`` as JPEG2000 with a target compression ratio ``cr``.

    Always validates the result strictly:
      - file exists
      - file size > 0
      - file starts with a real JP2 signature (or J2K SOC)
      - decoded image has the same HxWx3 shape and uint8 dtype as the input
    """
    if image_uint8.dtype != np.uint8 or image_uint8.ndim != 3 or image_uint8.shape[2] != 3:
        raise ValueError("encode_decode_jpeg2000 expects an HxWx3 uint8 RGB array")

    if _active_backend is None:
        check_jpeg2000_backend()
    assert _active_backend is not None
    name, encode, decode = _active_backend
    if backend_name is not None and backend_name != name:
        # Allow the caller to pin a backend, but warn rather than silently switching.
        warnings.warn(
            f"[jpeg2000] requested backend {backend_name!r} differs from active {name!r}",
            stacklevel=2,
        )

    compressed_path = (
        config.COMPRESSED_DIR / "jpeg2000" / f"{image_stem}__{level_name}.jp2"
    ).resolve()
    reconstructed_path = (
        config.RECONSTRUCTED_DIR / "jpeg2000" / f"{image_stem}__{level_name}.png"
    ).resolve()
    compressed_path.parent.mkdir(parents=True, exist_ok=True)
    reconstructed_path.parent.mkdir(parents=True, exist_ok=True)

    if compressed_path.exists():
        compressed_path.unlink()

    t0 = time.perf_counter()
    encode(image_uint8, compressed_path, int(cr))
    encoding_time_ms = (time.perf_counter() - t0) * 1000.0

    if not compressed_path.exists():
        raise RuntimeError(f"JPEG2000 encoding produced no file at {compressed_path}")
    if compressed_path.stat().st_size <= 0:
        raise RuntimeError(f"JPEG2000 file at {compressed_path} has zero size")
    if not _is_real_jp2_file(compressed_path):
        raise RuntimeError(
            f"JPEG2000 file at {compressed_path} is not a real JP2/J2K stream "
            "(missing signature). Refusing to silently save a non-JP2 file."
        )

    t0 = time.perf_counter()
    recon = decode(compressed_path)
    decoding_time_ms = (time.perf_counter() - t0) * 1000.0

    if recon.shape != image_uint8.shape:
        raise RuntimeError(
            f"JPEG2000 decoded shape {recon.shape} != original {image_uint8.shape}"
        )
    if recon.dtype != np.uint8:
        recon = recon.astype(np.uint8)

    save_png(recon, reconstructed_path)

    return {
        "compressed_path": str(compressed_path),
        "reconstructed_path": str(reconstructed_path),
        "encoding_time_ms": float(encoding_time_ms),
        "decoding_time_ms": float(decoding_time_ms),
        "backend": name,
        "reconstructed_array": recon,
    }
