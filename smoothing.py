"""
smoothing.py — a slew-rate limiter: caps how fast a value is allowed to
change per second, regardless of how fast the raw target jumps around.

This is what turns brake output from an abrupt 0-or-100 into something that
ramps like a real hydraulic/mechanical system would.
"""


class SlewRateLimiter:
    def __init__(self, max_change_per_second):
        self.max_change_per_second = max_change_per_second
        self.value = 0.0
        self._initialized = False

    def update(self, target, dt):
        if not self._initialized:
            self.value = target
            self._initialized = True
            return self.value

        max_delta = self.max_change_per_second * max(dt, 0.0)
        delta = target - self.value
        if delta > max_delta:
            delta = max_delta
        elif delta < -max_delta:
            delta = -max_delta

        self.value += delta
        return self.value

    def force(self, value):
        """
        Immediately snaps to value, bypassing the rate limit entirely — for
        safety overrides (e.g. the perception watchdog) that must not wait
        to ramp up. Subsequent normal update() calls slew from this new
        value like usual, so recovering from a forced 100% brake still
        releases smoothly rather than jumping straight back down.
        """
        self.value = value
        self._initialized = True
        return self.value
