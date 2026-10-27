using ArchenaAI.Common.MCP.Abstractions;
using System.Linq;

namespace ArchenaAI.Common.MCP.Models;

/// <summary>
/// Adapts a provider-neutral governance evaluation to the verdict contract.
/// The historical class name is retained to avoid an unnecessary public rename.
/// </summary>
public sealed class ArchitectureVerdictAdapter : IVerdict
{
    private readonly GovernanceEvaluationResult _result;
    private readonly GovernanceSeverity _denyThreshold;

    public ArchitectureVerdictAdapter(
        GovernanceEvaluationResult result,
        GovernanceSeverity denyThreshold)
    {
        _result = result ?? throw new ArgumentNullException(nameof(result));
        _denyThreshold = denyThreshold;
    }

    private IEnumerable<GovernanceRuleResult> Violations =>
        _result.RuleResults.Where(rule => !rule.IsCompliant);

    public bool IsAllowed => _result.IsAllowed(_denyThreshold);

    public GovernanceSeverity MaxSeverity =>
        Violations.Any()
            ? Violations.Max(rule => rule.Severity)
            : GovernanceSeverity.Info;

    public IReadOnlyCollection<string> ViolatedRuleIds =>
        Violations.Select(rule => rule.RuleId).ToArray();
}
