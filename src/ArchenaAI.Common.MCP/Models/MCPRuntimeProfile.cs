using ArchenaAI.Common.MCP.Abstractions;

namespace ArchenaAI.Common.MCP.Models;

public sealed class McpRuntimeProfile
{
    public string Name { get; init; } = string.Empty;
    public IReadOnlyCollection<string> EnabledRuleIds { get; init; } = Array.Empty<string>();
    public ExecutionBudget Budget { get; init; } = ExecutionBudget.ReadOnly();
    public IReadOnlyCollection<Capability> AllowedCapabilities { get; init; } = Array.Empty<Capability>();
    public GovernanceSeverity DenyThreshold { get; init; } = GovernanceSeverity.Critical;
}
