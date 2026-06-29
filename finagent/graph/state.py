"""Back-compat shim — agent state MOVED to `finagent.agents.state`.

Re-exported here so existing imports keep working, e.g.:
    from finagent.graph.state import AgentState, SubQueries, ...
Remove this shim once every consumer imports from `finagent.agents.state`.
"""

from finagent.agents.state import (  # noqa: F401
    AgentState,
    SubQueries,
    ClaimVerdict,
    CriticReport,
    ChunkScore,
    GraderReport,
    RewrittenQuery,
    QueryRoute,
    RouterReport,
    PlannedQuery,
    QueryPlan,
    TableTitle,
    PandasCode,
    NumericClaim,
    NumericVerification,
    MarketIntent,
    XBRLQuery,
    XBRLQueryBatch,
    CalcQuery,
    CalcQueryBatch,
    FormulaSpec,
    CorpusGateQuery,
    EdgarQuery,
)

__all__ = [
    "AgentState", "SubQueries", "ClaimVerdict", "CriticReport", "ChunkScore",
    "GraderReport", "RewrittenQuery", "QueryRoute", "RouterReport",
    "PlannedQuery", "QueryPlan", "TableTitle",
    "PandasCode", "NumericClaim", "NumericVerification", "MarketIntent",
    "XBRLQuery", "XBRLQueryBatch", "CalcQuery", "CalcQueryBatch", "FormulaSpec",
    "CorpusGateQuery", "EdgarQuery",
]
