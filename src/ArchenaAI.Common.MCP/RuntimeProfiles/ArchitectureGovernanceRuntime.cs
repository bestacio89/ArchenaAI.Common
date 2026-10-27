using ArchenaAI.Common.MCP.Abstractions;
using ArchenaAI.Common.MCP.Models;

namespace ArchenaAI.Common.MCP.RuntimeProfiles;

public static class ArchitectureGovernanceRuntime
{
    public static readonly McpRuntimeProfile Instance = new()
    {
        Name = "ArchitectureGovernanceRuntime",
        EnabledRuleIds = ["AEG-ARCH-*", "AEG-DES-*", "AEG-DEP-*"],
        Budget = ExecutionBudget.Architecture(),
        AllowedCapabilities = [Capabilities.ReadRepository(), Capabilities.GenerateCodeArtifacts(), Capabilities.ProposeRefactorings()],
        DenyThreshold = GovernanceSeverity.Critical
    };
}
