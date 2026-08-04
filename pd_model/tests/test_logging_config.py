"""Tests for pd_model.logging_config — PIIRedactingFilter and install_pii_filter."""

from __future__ import annotations

import io
import logging

from pd_model.logging_config import PIIRedactingFilter, install_pii_filter


def _make_record(msg: str, args: tuple = ()) -> logging.LogRecord:
    return logging.LogRecord(
        name="test.pii",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=msg,
        args=args,
        exc_info=None,
    )


class TestPIIRedactingFilter:
    def test_uganda_international_redacted(self):
        f = PIIRedactingFilter()
        r = _make_record("agent %s", ("256701234567",))
        f.filter(r)
        assert "256701234567" not in r.getMessage()
        assert "[MSISDN]" in r.getMessage()

    def test_kenya_international_redacted(self):
        # Kenya country code 254 was NOT matched by the old regex — regression guard.
        f = PIIRedactingFilter()
        r = _make_record("agent %s", ("254712345678",))
        f.filter(r)
        assert "254712345678" not in r.getMessage()
        assert "[MSISDN]" in r.getMessage()

    def test_plus_prefix_international_redacted(self):
        f = PIIRedactingFilter()
        r = _make_record("+256701234567 processed")
        f.filter(r)
        assert "+256701234567" not in r.getMessage()
        assert "[MSISDN]" in r.getMessage()

    def test_local_format_redacted(self):
        f = PIIRedactingFilter()
        r = _make_record("local number %s", ("0701234567",))
        f.filter(r)
        assert "0701234567" not in r.getMessage()
        assert "[MSISDN]" in r.getMessage()

    def test_non_pii_digits_not_redacted(self):
        f = PIIRedactingFilter()
        r = _make_record("count=%s rows=12345", (42,))
        f.filter(r)
        msg = r.getMessage()
        assert "42" in msg
        assert "12345" in msg

    def test_dict_arg_with_msisdn_redacted(self):
        # Dict embedded as %s arg — getMessage() formats it, then regex applies.
        f = PIIRedactingFilter()
        r = _make_record("%s", ({"msisdn": "256701234567"},))
        f.filter(r)
        assert "256701234567" not in r.getMessage()

    def test_args_cleared_after_filter(self):
        # Prevents double-formatting if a downstream handler re-formats the record.
        f = PIIRedactingFilter()
        r = _make_record("x=%s", ("hello",))
        f.filter(r)
        assert r.args == ()

    def test_multiple_msisdns_in_one_message(self):
        f = PIIRedactingFilter()
        r = _make_record("from=%s to=%s", ("256701234567", "254712345678"))
        f.filter(r)
        msg = r.getMessage()
        assert "256701234567" not in msg
        assert "254712345678" not in msg
        assert msg.count("[MSISDN]") == 2

    def test_non_msisdn_254_prefix_not_redacted(self):
        # A short number starting with 254 that is not a valid MSISDN (too few digits)
        f = PIIRedactingFilter()
        r = _make_record("code %s", ("2541",))
        f.filter(r)
        assert "2541" in r.getMessage()


class TestInstallPiiFilter:
    def test_filter_installed_on_root_handler(self):
        root = logging.getLogger()
        handler = logging.StreamHandler(io.StringIO())
        root.addHandler(handler)
        try:
            install_pii_filter()
            assert any(isinstance(f, PIIRedactingFilter) for f in handler.filters)
        finally:
            root.removeHandler(handler)

    def test_idempotent_no_duplicate_filters(self):
        root = logging.getLogger()
        handler = logging.StreamHandler(io.StringIO())
        root.addHandler(handler)
        try:
            install_pii_filter()
            install_pii_filter()
            pii_count = sum(1 for f in handler.filters if isinstance(f, PIIRedactingFilter))
            assert pii_count == 1
        finally:
            root.removeHandler(handler)

    def test_propagating_logger_scrubbed_via_root(self):
        # Simulate an extrafloat-style logger: bare getLogger, propagate=True.
        sink = io.StringIO()
        root = logging.getLogger()
        handler = logging.StreamHandler(sink)
        handler.setLevel(logging.DEBUG)
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        try:
            install_pii_filter()
            child = logging.getLogger("extrafloat.test_child")
            child.propagate = True
            child.setLevel(logging.DEBUG)
            child.info("Processing agent %s", "256701234567")
            output = sink.getvalue()
            assert "256701234567" not in output
            assert "[MSISDN]" in output
        finally:
            root.removeHandler(handler)
            root.setLevel(logging.WARNING)
