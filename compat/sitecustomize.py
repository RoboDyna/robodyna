"""Python 3.10 compatibility shim for the vendored openpi.

RoboDyna ships openpi under policy/pi05/src, which uses ``datetime.UTC``
(added in Python 3.11). The simulator environment uses Python 3.10, so running
policy/pi05 in-process fails on this one attribute.

Put this directory first on PYTHONPATH to apply the shim at interpreter startup
(sitecustomize is imported automatically) without editing openpi's source.
"""
import datetime as _datetime

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc
