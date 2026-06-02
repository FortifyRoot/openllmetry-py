# NOTE:
# This file has been modified by FortifyRoot.
# Original source: https://github.com/traceloop/openllmetry

from typing import Callable, Optional


class Config:
    exception_logger = None
    use_legacy_attributes = True
    upload_base64_image: Optional[Callable[[str, str, str, str], str]] = None
