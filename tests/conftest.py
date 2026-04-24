# Author: Bradley R. Kinnard
"""
Root pytest conftest.

Disables NLI model downloads during test runs. The NLI fallback pulls a
~500MB transformer model on first use, which is unnecessary for CI and
masks rule-based coverage. Tests that need NLI explicitly should
re-enable it locally.
"""

import pytest


@pytest.fixture(autouse=True, scope="session")
def _disable_nli_model_loading():
    from backend.core.bel import nli_detector

    nli_detector._nli_available = False
    yield
