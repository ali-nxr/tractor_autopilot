"""
playback — play a recorded Intel RealSense sequence through a perception
pipeline that was written for a live camera, with no changes to that
pipeline.

Self-contained by design: nothing in here imports the host project. The
host supplies a PlaybackOptions, optionally injects its own IMU fusion
object, and gets back a frame source shaped exactly like a live
RealSenseCapture. See README.md in this folder for how to reuse it.

    from playback import (PlaybackOptions, RealSenseBagSource,
                          PlaybackController, PlaybackCapturePipeline,
                          select_sequence)

    info = select_sequence(r"C:\\data\\recordings")
    options = PlaybackOptions(path=info.path, want_motion=info.has_motion)
    source = RealSenseBagSource(options, info=info)
    controller = PlaybackController(source, on_discontinuity=reset_my_state)
    pipeline = PlaybackCapturePipeline(controller)
    pipeline.start()

    frames, capture_time = pipeline.get_latest()   # live-compatible
"""

from .options import PlaybackOptions
from .discovery import (
    SEQUENCE_EXTENSIONS,
    SequenceInfo,
    StreamInfo,
    list_sequence_files,
    list_sequences,
    probe_sequence,
    select_sequence,
)
from .bag_source import PlaybackEnded, RealSenseBagSource
from .controller import PlaybackController
from .pipeline import PlaybackCapturePipeline
from .imu_adapter import playback_fusion, supports_recorded_timebase

__all__ = [
    "PlaybackOptions",
    "SEQUENCE_EXTENSIONS",
    "SequenceInfo",
    "StreamInfo",
    "list_sequence_files",
    "list_sequences",
    "probe_sequence",
    "select_sequence",
    "PlaybackEnded",
    "RealSenseBagSource",
    "PlaybackController",
    "PlaybackCapturePipeline",
    "playback_fusion",
    "supports_recorded_timebase",
]
