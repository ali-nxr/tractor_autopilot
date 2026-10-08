"""
playback/discovery.py — find recorded RealSense sequences in a folder and
read back what is actually inside them, WITHOUT starting playback.

Two separate questions get answered here, and the difference matters:

  1. What streams does the recording ADVERTISE? That comes from the
     RealSense SDK itself (context.load_device), so it works for any
     container the installed SDK can open.

  2. What streams does the recording actually CONTAIN MESSAGES for? Not
     the same thing at all. A real recording in this project's own data
     set (20261007_145608) advertises a Motion Module with accel+gyro
     profiles but holds ZERO motion messages — and a pipeline that
     enables accel/gyro against it then waits forever for motion frames
     that never arrive and yields no framesets at all, silently. For the
     rosbag2/sqlite container (.db3) the per-topic message counts can be
     read directly, so that trap is visible before playback starts.

Question 2 can only be answered for containers understood here, so
RealSenseBagSource never relies on it — it has its own runtime stream
ladder for exactly this case. What is read here is used to pick sensible
defaults and to tell the user what they are about to play.
"""

import os
import sqlite3
from dataclasses import dataclass, field

import pyrealsense2 as rs

# Containers the RealSense SDK can play back. .db3 is the rosbag2/sqlite
# container the current SDK records to; .bag is the older ROS1 one.
SEQUENCE_EXTENSIONS = (".db3", ".bag")


@dataclass
class StreamInfo:
    """One stream a recording advertises."""
    stream_type: str                  # "depth" | "color" | "infrared" | "accel" | "gyro" | ...
    index: int
    format_name: str
    fps: int
    width: int = 0                    # 0 for motion streams
    height: int = 0
    message_count: int = None         # None = could not be determined

    @property
    def is_video(self):
        return self.width > 0 and self.height > 0

    @property
    def has_data(self):
        """True, False, or None when unknown — never guess False from ignorance."""
        if self.message_count is None:
            return None
        return self.message_count > 0

    def describe(self):
        geom = f"{self.width}x{self.height}" if self.is_video else "-"
        count = "?" if self.message_count is None else str(self.message_count)
        return (f"{self.stream_type}({self.index}) {geom} @{self.fps}fps "
                f"{self.format_name}  msgs={count}")


@dataclass
class SequenceInfo:
    """Everything known about one recorded sequence before playing it."""
    path: str
    name: str
    size_bytes: int = 0
    duration_s: float = 0.0
    device_name: str = ""
    streams: list = field(default_factory=list)
    error: str = None

    def stream(self, stream_type):
        for s in self.streams:
            if s.stream_type == stream_type:
                return s
        return None

    def _usable(self, stream_type):
        """Advertised, and not known to be empty."""
        s = self.stream(stream_type)
        return s is not None and s.has_data is not False

    @property
    def has_color(self):
        return self._usable("color")

    @property
    def has_depth(self):
        return self._usable("depth")

    @property
    def has_motion(self):
        """Accel AND gyro both present with data (or with unknown counts)."""
        return self._usable("accel") and self._usable("gyro")

    @property
    def has_ir(self):
        return self._usable("infrared")

    @property
    def video_frame_count(self):
        s = self.stream("color") or self.stream("depth")
        return None if s is None else s.message_count

    def summary(self):
        if self.error:
            return f"{self.name} - UNREADABLE: {self.error}"
        mb = self.size_bytes / (1024 * 1024)
        parts = [self.name, f"{self.duration_s:.1f}s", f"{mb:.0f} MB"]
        if self.device_name:
            parts.append(self.device_name)
        n = self.video_frame_count
        if n:
            parts.append(f"{n} frames")
        parts.append("IMU" if self.has_motion else "no IMU")
        if self.has_ir:
            parts.append("IR")
        return "  |  ".join(parts)


def _rs_stream_type_name(stream_type):
    """rs.stream.color -> "color"."""
    return str(stream_type).split(".")[-1]


def _rs_format_name(fmt):
    """rs.format.rgb8 -> "rgb8"."""
    return str(fmt).split(".")[-1]


def _message_counts_db3(path):
    """
    Per-stream message counts straight out of the rosbag2/sqlite
    container, keyed by (lowercase stream name, index).

    Returns None when the container could not be read at all, and a dict
    when it could. That distinction carries real meaning and must not be
    collapsed to an empty dict: a stream MISSING from a dict that was
    read successfully genuinely has no data, whereas None means nothing
    is known either way. Treating "unreadable" as "everything is empty"
    would make every .bag file look like it had no streams.

    Topic names inside a RealSense recording look like
    "/device_0/sensor_1/Color_0/image/data" or
    "/device_0/sensor_2/Gyro_0/imu/data", so the stream label is the
    "<Name>_<index>" path segment two levels above the trailing "data".

    Never raises.
    """
    if not path.lower().endswith(".db3"):
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        try:
            rows = conn.execute(
                "SELECT t.name, COUNT(m.id) FROM topics t "
                "LEFT JOIN messages m ON m.topic_id = t.id "
                "WHERE t.name LIKE '%/data' GROUP BY t.id"
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return None

    counts = {}
    for topic, count in rows:
        segments = topic.strip("/").split("/")
        if len(segments) < 3 or segments[-1] != "data":
            continue
        label = segments[-3]          # Color_0 / Depth_0 / Gyro_0 / Accel_0
        name, _, idx = label.rpartition("_")
        if not name:
            continue
        try:
            idx = int(idx)
        except ValueError:
            continue
        counts[(name.lower(), idx)] = int(count)
    return counts


def probe_sequence(path):
    """
    Open `path` with the RealSense SDK just long enough to read its
    duration and stream list, then unload it again. Always returns a
    SequenceInfo — a failure is reported in .error rather than raised, so
    one bad file in a folder never breaks listing the rest.
    """
    path = os.path.abspath(path)
    info = SequenceInfo(path=path, name=os.path.basename(path))
    try:
        info.size_bytes = os.path.getsize(path)
    except OSError:
        pass

    counts = _message_counts_db3(path)
    ctx = rs.context()
    loaded = False
    try:
        device = ctx.load_device(path)
        loaded = True
        try:
            info.device_name = device.get_info(rs.camera_info.name)
        except Exception:
            pass
        try:
            info.duration_s = device.as_playback().get_duration().total_seconds()
        except Exception:
            pass

        for sensor in device.sensors:
            for profile in sensor.profiles:
                stream_type = _rs_stream_type_name(profile.stream_type())
                index = profile.stream_index()
                entry = StreamInfo(
                    stream_type=stream_type,
                    index=index,
                    format_name=_rs_format_name(profile.format()),
                    fps=profile.fps(),
                    # Counts read, but nothing for this stream => it is
                    # advertised and empty (real case: 20261007_145608's
                    # accel/gyro). Counts unreadable => leave it unknown.
                    message_count=(None if counts is None
                                   else counts.get((stream_type, index), 0)),
                )
                try:
                    video = profile.as_video_stream_profile()
                    entry.width, entry.height = video.width(), video.height()
                except Exception:
                    pass
                info.streams.append(entry)
    except Exception as e:
        info.error = f"{type(e).__name__}: {e}"
    finally:
        if loaded:
            # Must happen, or this path stays claimed inside the SDK
            # context and the pipeline that plays it back later can
            # collide with this probe.
            try:
                ctx.unload_device(path)
            except Exception:
                pass
    return info


def list_sequence_files(folder):
    """Recorded-sequence file paths in `folder`, name-sorted (the
    recorder timestamp-names them, so that is chronological)."""
    if not os.path.isdir(folder):
        return []
    names = [n for n in os.listdir(folder)
             if n.lower().endswith(SEQUENCE_EXTENSIONS)]
    return [os.path.join(folder, n) for n in sorted(names)]


def list_sequences(folder):
    """probe_sequence() for every recorded sequence in `folder`."""
    return [probe_sequence(p) for p in list_sequence_files(folder)]


def select_sequence(folder, selector=None, prefer="longest"):
    """
    Resolve a user's sequence choice to one SequenceInfo.

    selector:
      None             -> pick automatically, per `prefer`
      a path           -> that file (need not be inside `folder`)
      1-3 digits       -> index into the folder listing (0, 1, 2, ...)
      any other text   -> case-insensitive substring match on the file name

    The digit rule is length-limited on purpose. These recordings are
    timestamp-named, so a perfectly natural selector like "145904" is all
    digits — treating that as an index would send the user to a
    nonexistent entry [145904] instead of the file they plainly named.
    Indices are single- or double-digit in any real folder, and an
    in-range index is also tried as a name match first, so the two
    meanings cannot collide silently.

    prefer: "longest" (most data to look at — the useful default),
            "first", or "last".

    Raises FileNotFoundError / ValueError whose message names the actual
    candidates, so a typo is self-diagnosing.
    """
    if selector is not None and os.path.isfile(str(selector)):
        return probe_sequence(str(selector))

    sequences = list_sequences(folder)
    if not sequences:
        raise FileNotFoundError(
            f"No recorded sequences ({', '.join(SEQUENCE_EXTENSIONS)}) found in {folder!r}"
        )

    if selector is None:
        playable = [s for s in sequences if not s.error] or sequences
        if prefer == "first":
            return playable[0]
        if prefer == "last":
            return playable[-1]
        return max(playable, key=lambda s: s.duration_s)

    selector = str(selector).strip()
    matches = [s for s in sequences if selector.lower() in s.name.lower()]
    if len(matches) == 1:
        return matches[0]

    if selector.isdigit() and len(selector) <= 3:
        index = int(selector)
        if 0 <= index < len(sequences):
            return sequences[index]

    listing = "\n".join(f"  [{i}] {s.summary()}" for i, s in enumerate(sequences))
    if len(matches) > 1:
        names = "\n".join(f"  {s.name}" for s in matches)
        raise ValueError(f"{selector!r} matches more than one sequence:\n{names}")
    raise ValueError(f"No recorded sequence matching {selector!r}. Available:\n{listing}")
