"""Agent core: parent graph, nodes, state, runner and the ReAct subgraph.

Deliberately import-free: submodules are imported directly
(``askanswer.core.graph``, ``askanswer.core.runner`` …) so that lower layers such
as :mod:`askanswer.intents` can depend on :mod:`askanswer.core.state` without
pulling in the whole graph.
"""
