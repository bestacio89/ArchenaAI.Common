using ArchenaAI.Common.MCP.Abstractions;
using ArchenaAI.Common.MCP.Models;

namespace ArchenaAI.Common.MCP.RuntimeProfiles;

public static class HighRiskControlledRuntime
{
    public static readonly McpRuntimeProfile Instance = new()
    {
        Name = "HighRiskControlledRuntime",
        EnabledRuleIds = ["AEG-ARCH-*", "AEG-PERSIST-*", "AEG-SEC-*", "AEG-PERF-*"],
        Budget = ExecutionBudget.Migration(),
        AllowedCapabilities = [Capabilities.DbWriteTransactional(), Capabilities.ModifySchema(), Capabilities.EmitAuditEvents()],
        DenyThreshold = GovernanceSeverity.Blocker
    };
}
