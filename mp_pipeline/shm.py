"""
mp_pipeline/shm.py — shared-memory buffers that let full-resolution frames
move between processes without being pickled or copied through a pipe.

FrameRing: N slots, each holding one captured frame (color, depth, xyz,
raw-valid mask, IR). Ownership is passed by SLOT INDEX through ordinary
multiprocessing queues: the capture process takes a free index, fills that
slot, and hands the index down the pipeline; the last stage (render) gives
it back. A slot is therefore only ever written by one process at a time —
no locks needed on the arrays themselves.

DisplayRing: a few output buffers (annotated HUD frame + debug mosaic) the
render process writes and the UI process reads. The UI may lag behind, so
each buffer carries a sequence number written AFTER the pixels (a seqlock):
the reader checks it before and after copying and discards a copy that was
overwritten mid-read.
"""

from multiprocessing import shared_memory

import numpy as np


def _layout(spec):
    """spec: {name: (shape, dtype)} -> ({name: (offset, shape, dtype)}, total bytes)."""
    offset, out = 0, {}
    for name, (shape, dtype) in spec.items():
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        offset = (offset + 63) // 64 * 64          # cache-line align each array
        out[name] = (offset, tuple(shape), dtype.str)
        offset += nbytes
    return out, offset


class _Block:
    """One shared-memory block holding `count` copies of a spec'd set of arrays."""

    def __init__(self, spec, count, name=None, create=False):
        self.layout, self.stride = _layout(spec)
        self.stride = (self.stride + 63) // 64 * 64
        self.count = count
        self._spec = spec
        if create:
            self.shm = shared_memory.SharedMemory(create=True, size=max(1, self.stride * count))
        else:
            self.shm = shared_memory.SharedMemory(name=name)
        self._owner = create
        self._views = [{
            key: np.ndarray(shape, dtype=np.dtype(dt), buffer=self.shm.buf, offset=i * self.stride + off)
            for key, (off, shape, dt) in self.layout.items()
        } for i in range(count)]

    def handle(self):
        """Picklable description another process can attach() with."""
        return {"name": self.shm.name, "spec": self._spec, "count": self.count}

    def __getitem__(self, i):
        return self._views[i]

    def close(self):
        # Views must be dropped before the mapping can be closed.
        self._views = []
        try:
            self.shm.close()
        except Exception:
            pass
        if self._owner:
            try:
                self.shm.unlink()
            except Exception:
                pass


class FrameRing(_Block):
    """Captured frames. See module docstring for the ownership protocol."""

    @staticmethod
    def spec_for(height, width):
        return {
            "color": ((height, width, 3), np.uint8),
            "depth": ((height, width), np.float32),
            # no xyz: it is deprojected downstream at processing resolution
            # (perception_core.deproject_proc) — 11 MB less per frame to fill
            "raw_valid": ((height, width), np.bool_),
            "ir": ((height, width), np.uint8),
        }

    @classmethod
    def create(cls, height, width, count):
        return cls(cls.spec_for(height, width), count, create=True)

    @classmethod
    def attach(cls, handle):
        return cls(handle["spec"], handle["count"], name=handle["name"])


class DisplayRing(_Block):
    """Render -> UI output buffers, seqlock-protected (see module docstring)."""

    @staticmethod
    def spec_for(height, width, debug_shape):
        spec = {
            "seq": ((1,), np.int64),
            "annotated": ((height, width, 3), np.uint8),
        }
        if debug_shape is not None:
            spec["debug"] = (tuple(debug_shape), np.uint8)
        return spec

    @classmethod
    def create(cls, height, width, debug_shape, count=4):
        ring = cls(cls.spec_for(height, width, debug_shape), count, create=True)
        for i in range(count):
            ring[i]["seq"][0] = -1
        return ring

    @classmethod
    def attach(cls, handle):
        return cls(handle["spec"], handle["count"], name=handle["name"])

    def write(self, index, seq, annotated, debug=None):
        buf = self[index]
        buf["seq"][0] = -1                     # mark "being written"
        buf["annotated"][...] = annotated
        if debug is not None and "debug" in buf:
            buf["debug"][...] = debug
        buf["seq"][0] = seq                    # publish

    def read(self, index, seq):
        """Copies of (annotated, debug-or-None), or None if the buffer was
        overwritten while (or before) being read."""
        buf = self[index]
        if buf["seq"][0] != seq:
            return None
        annotated = buf["annotated"].copy()
        debug = buf["debug"].copy() if "debug" in buf else None
        if buf["seq"][0] != seq:
            return None
        return annotated, debug
