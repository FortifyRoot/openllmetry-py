# NOTE:
# This file has been modified by FortifyRoot.
# Original source: https://github.com/traceloop/openllmetry

from typing import Callable, Optional


class Config:
    exception_logger = None
    use_legacy_attributes = True
    # Set by instrumentor only when a real image uploader is provided (e.g.,
    # Traceloop cloud).  None by default — image processing is skipped when no
    # uploader is configured, avoiding KeyError on non-Traceloop endpoints.
    upload_base64_image: Optional[Callable[[str, str, str, str], str]] = None
