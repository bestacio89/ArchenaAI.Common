using ArchenaAI.Common.MCP.Abstractions;
using ArchenaAI.Common.MCP.Models;

namespace ArchenaAI.Common.MCP.RuntimeProfiles;

public static class PersistenceGovernedRuntime
{
    public static readonly McpRuntimeProfile Instance = new()
    {
        Name = "PersistenceGovernedRuntime",
        EnabledRuleIds = ["AEG-PERSIST-*", "AEG-SEC-*", "AEG-PERF-EXEC001"],
        Budget = ExecutionBudget.Persistence(),
        AllowedCapabilities = [Capabilities.DbRead(), Capabilities.DbWriteTransactional()],
        DenyThreshold = GovernanceSeverity.Critical
    };
}
