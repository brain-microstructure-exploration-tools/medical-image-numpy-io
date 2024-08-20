from __future__ import annotations

import importlib.metadata

import medical_image_numpy_io as m


def test_version():
    assert importlib.metadata.version("medical_image_numpy_io") == m.__version__
