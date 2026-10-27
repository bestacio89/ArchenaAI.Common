using ArchenaAI.Common.MCP.Abstractions;

namespace ArchenaAI.Common.MCP.Models;

/// <summary>
/// A provider-neutral, auditable result from one governance rule.
/// </summary>
public sealed record GovernanceRuleResult
{
    public required string RuleId { get; init; }
    public required string RuleName { get; init; }
    public required GovernanceSeverity Severity { get; init; }
    public required bool IsCompliant { get; init; }
    public required string Message { get; init; }
    public string? ActionId { get; init; }
    public string? Category { get; init; }
    public string? OriginComponent { get; init; }
    public DateTimeOffset EvaluatedAt { get; init; } = DateTimeOffset.UtcNow;
}
