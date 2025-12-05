public enum AgentMessageType
{
    Unknown = 0,

    // Core
    Request,
    Think,
    Act,
    Response,

    // Memory
    MemoryIndex,
    MemoryStore,
    MemorySearch,
    Normalize,
    Documentation,

    // DevOps
    DevOpsTask,

    // SRE
    Incident,
    LogAnalysis,

    // Governance
    ComplianceCheck,
    PolicyEvaluation
}
