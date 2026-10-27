namespace ArchenaAI.Common.MCP.Abstractions;

/// <summary>
/// Provider-neutral severity levels used by runtime governance decisions.
/// Higher values represent stronger reasons to reject execution.
/// </summary>
public enum GovernanceSeverity
{
    Info = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
    Blocker = 5
}
