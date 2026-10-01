"""Python 3.10 compatibility shim for the vendored openpi.

RoboDyna ships openpi under policy/pi05/src, which uses ``datetime.UTC`` --
added in 3.11. The only environment that has sapien + curobo + openpi together
is the RMBench conda env, and that is Python 3.10, so openpi fails to import
there on this one attribute and nothing else.

Putting this directory first on PYTHONPATH makes Python apply the shim at
interpreter startup (sitecustomize is imported automatically), which avoids
editing openpi's source.
"""
import datetime as _datetime

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc
