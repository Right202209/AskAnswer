"""Model layer: the hot-swappable global model proxy and role-based routing.

- :mod:`.proxy` – ``model`` (a lazily built, hot-swappable ``_ModelProxy``),
  ``set_model``, ``current_model_label`` and audit/telemetry callback injection;
- :mod:`.routing` – ``model_for(role)`` with per-role models and cross-provider
  fallback chains (``ROLE_ANSWER`` / ``ROLE_CLASSIFY`` / ``ROLE_EVALUATE`` /
  ``ROLE_SUMMARIZE``).
"""
