namespace ArchenaAI.Common.MCP.Models;

/// <summary>
/// Provider-neutral outcome of evaluating a proposed runtime action.
/// </summary>
public sealed class GovernanceEvaluationResult
{
    private readonly List<GovernanceRuleResult> _ruleResults = [];

    public required string Source { get; init; }
    public required string Target { get; init; }
    public string Category { get; set; } = "Execution";
    public IReadOnlyList<GovernanceRuleResult> RuleResults => _ruleResults;

    public bool IsAllowed(GovernanceSeverity denyThreshold) =>
        !_ruleResults.Any(result => !result.IsCompliant && result.Severity >= denyThreshold);

    public void AddRuleResult(GovernanceRuleResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        _ruleResults.Add(result);
    }
}
