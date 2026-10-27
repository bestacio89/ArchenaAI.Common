using ArchenaAI.Common.MCP.Abstractions;
using ArchenaAI.Common.MCP.Models;

namespace ArchenaAI.Common.MCP.RuntimeProfiles;

public static class AnalysisRuntime
{
    public static readonly McpRuntimeProfile Instance = new()
    {
        Name = "AnalysisRuntime",
        EnabledRuleIds = ["AEG-ARCH-*", "AEG-DEP-*", "AEG-NAME-*", "AEG-SEC-AUTH001", "AEG-SEC-SECRET001"],
        Budget = ExecutionBudget.ReadOnly(),
        AllowedCapabilities = [Capabilities.ReadRepository(), Capabilities.ReadMetadata(), Capabilities.InspectArchitecture()],
        DenyThreshold = GovernanceSeverity.Critical
    };
}
