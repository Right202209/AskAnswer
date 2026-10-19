"""Import smoke test: importing askanswer must have no side effects.

Run from an empty directory with a fresh HOME (see .github/workflows/ci.yml).
Asserts the process environment is unchanged and no state.db was created.
"""

import os

before = dict(os.environ)
import askanswer  # noqa: E402

assert dict(os.environ) == before, "environment changed on import"
assert not os.path.exists(os.path.expanduser("~/.askanswer/state.db")), "state.db created"
print("SMOKE OK", sorted(askanswer.__all__))
