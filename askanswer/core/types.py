"""纯领域类型：不依赖 LangChain / LangGraph（架构文档 6.2）。"""

from __future__ import annotations

from typing import Literal

# 流程阶段：取代过去的自由字符串
Phase = Literal["understood", "tool_called", "completed", "retry_search"]
PHASE_UNDERSTOOD: Phase = "understood"      # understand 刚完成分类（本轮首次进入 answer）
PHASE_TOOL_CALLED: Phase = "tool_called"    # answer 产出 tool_calls，等待工具执行
PHASE_COMPLETED: Phase = "completed"        # 已产出最终答案
PHASE_RETRY: Phase = "retry_search"         # review 判定需要重试
PHASES: frozenset[str] = frozenset(
    {PHASE_UNDERSTOOD, PHASE_TOOL_CALLED, PHASE_COMPLETED, PHASE_RETRY}
)


# 工具风险等级与能力（架构文档 7.1 / 7.2）
Risk = Literal["read", "network", "write", "exec", "paid", "external"]
RISK_READ: Risk = "read"            # 只读本地
RISK_NETWORK: Risk = "network"      # 只读网络（必须有超时）
RISK_WRITE: Risk = "write"          # 写本地文件
RISK_EXEC: Risk = "exec"            # 执行命令
RISK_PAID: Risk = "paid"            # 付费外部 API
RISK_EXTERNAL: Risk = "external"    # 第三方 MCP
# 这三类有副作用：注册时必须声明确认类，否则拒绝注册（不可绕过的不变量）
RISKS_REQUIRING_APPROVAL = frozenset({RISK_WRITE, RISK_EXEC, RISK_PAID})

CAP_UTIL = "util"
CAP_NET_LOOKUP = "net.lookup"
CAP_WEB_SEARCH = "web.search"
CAP_FS_READ = "fs.read"
CAP_FS_WRITE = "fs.write"
CAP_EXEC_SHELL = "exec.shell"
CAP_DATA_SQL = "data.sql"
CAP_AGENT_SPEC = "agent.spec"
CAP_AGENT_RESEARCH = "agent.research"
CAP_AGENT_DECISION = "agent.decision"
CAP_MCP = "mcp"
