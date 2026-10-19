"""Observability: local audit trail, argument redaction and external telemetry.

Three independent write paths share one minimal event schema
(kind / tool_name / duration_ms / tokens):

- :mod:`askanswer.observability.audit` – run-scoped audit buffer in ``contextvars``,
  flushed to the persistence backend by :func:`askanswer.run_leg`;
- :mod:`askanswer.observability.telemetry` – LangSmith / OpenTelemetry exporters,
  zero-cost and SDK-free when not enabled;
- :mod:`askanswer.observability.redaction` – masking applied to audited tool args.

Submodules are imported lazily by callers; importing this package stays cheap.
"""
