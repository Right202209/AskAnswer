"""按表划分的仓储。"""

from askanswer.storage.repos.audit import AuditRepo
from askanswer.storage.repos.labels import LabelRepo
from askanswer.storage.repos.threads import ThreadRepo

__all__ = ["AuditRepo", "LabelRepo", "ThreadRepo"]
