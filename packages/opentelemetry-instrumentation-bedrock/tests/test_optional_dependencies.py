"""Tests for optional dependency behavior."""

from __future__ import annotations

import builtins
import importlib
import sys


def test_bedrock_span_utils_import_without_anthropic(monkeypatch):
    """Bedrock instrumentation import should not require the Anthropic package."""
    module_name = "opentelemetry.instrumentation.bedrock.span_utils"
    sys.modules.pop(module_name, None)
    sys.modules.pop("anthropic", None)

    original_import = builtins.__import__

    def import_without_anthropic(name, *args, **kwargs):
        if name == "anthropic":
            raise ModuleNotFoundError("No module named 'anthropic'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_anthropic)

    module = importlib.import_module(module_name)

    assert module is not None
