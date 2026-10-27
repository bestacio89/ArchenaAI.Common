using ArchenaAI.Common.MCP.Abstractions;
using ArchenaAI.Common.MCP.Models;

namespace ArchenaAI.Common.MCP.RuntimeProfiles;

public static class AgentExecutionRuntime
{
    public static readonly McpRuntimeProfile Instance = new()
    {
        Name = "AgentExecutionRuntime",
        EnabledRuleIds = ["AEG-ARCH-*", "AEG-DES-*", "AEG-DEP-*", "AEG-PERF-*", "AEG-SEC-*"],
        Budget = ExecutionBudget.Agent(),
        AllowedCapabilities = [Capabilities.ReadRepository(), Capabilities.CallInternalServices(), Capabilities.EmitEvents()],
        DenyThreshold = GovernanceSeverity.Critical
    };
}
