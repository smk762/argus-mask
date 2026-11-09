from __future__ import annotations

import argus_mask


def test_version_is_exposed() -> None:
    assert isinstance(argus_mask.__version__, str)
    assert argus_mask.__version__
