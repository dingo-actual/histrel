#!/usr/bin/env python3
"""
Clock utilities for the benchmark agent.

Provides monotonic and UTC clock helpers for timing measurement and
correlation. Monotonic clock is used for durations; UTC timestamps are
used for correlation and journal records.

    Important Functions:
        monotonic_now() -> float: Monotonic seconds since an arbitrary epoch.
        utc_now() -> float: UTC seconds since the Unix epoch.
        utc_timestamp() -> int: Integer UTC seconds (Unix timestamp).
"""

from __future__ import annotations

import time


def monotonic_now() -> float:
    """Return monotonic clock seconds for duration measurement.

    Returns:
        (float) Monotonic seconds, guaranteed non-decreasing.
    """
    return time.monotonic()


def utc_now() -> float:
    """Return current UTC time as seconds since the Unix epoch.

    Returns:
        (float) UTC seconds since epoch.
    """
    return time.time()


def utc_timestamp() -> int:
    """Return current UTC time as an integer Unix timestamp in seconds.

    Returns:
        (int) Integer Unix timestamp.
    """
    return int(time.time())
