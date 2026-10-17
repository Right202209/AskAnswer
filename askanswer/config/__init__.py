"""类型化配置（架构文档 P3 / D6）。

* :mod:`askanswer.config.schema` —— Pydantic ``Settings``，字段有类型与校验
* :mod:`askanswer.config.loader` —— 唯一读取 ``ASKANSWER_*`` 等环境变量的地方
* :mod:`askanswer.config.paths`  —— 数据目录约定（state.db / history / mcp.json）

分层来源（进程 env > local > project > user > .env）仍由 ``askanswer.settings``
负责装载；业务模块只拿 ``Settings`` 对象，不再各自 ``os.environ.get``。
"""

from .loader import current_settings, load_settings
from .schema import ModelRoles, Settings

__all__ = ["ModelRoles", "Settings", "current_settings", "load_settings"]
