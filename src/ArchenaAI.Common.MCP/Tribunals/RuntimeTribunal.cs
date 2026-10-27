using ArchenaAI.Common.MCP.Abstractions;
using ArchenaAI.Common.MCP.Models;

namespace ArchenaAI.Common.MCP.Tribunals;

/// <summary>
/// Runs only the policy rules enabled by the supplied runtime profile.
/// A violating result at or above the profile's deny threshold stops evaluation.
/// </summary>
public sealed class RuntimeTribunal
{
    private readonly IReadOnlyList<IPolicyEvaluator> _policies;

    public RuntimeTribunal(IEnumerable<IPolicyEvaluator> policies)
    {
        _policies = policies?.ToArray() ?? throw new ArgumentNullException(nameof(policies));
    }

    public GovernanceEvaluationResult Evaluate(
        IRuntimeActionDescriptor action,
        IAuthorityToken authority,
        McpRuntimeProfile runtimeProfile)
    {
        ArgumentNullException.ThrowIfNull(action);
        ArgumentNullException.ThrowIfNull(authority);
        ArgumentNullException.ThrowIfNull(runtimeProfile);

        var result = RuntimeEvaluatorResult.ForAction(action);
        result.Category = runtimeProfile.Name;

        foreach (var policy in _policies)
        {
            if (!runtimeProfile.EnabledRuleIds.Any(id =>
                    id.EndsWith('*')
                        ? policy.RuleId.StartsWith(id[..^1], StringComparison.Ordinal)
                        : string.Equals(policy.RuleId, id, StringComparison.Ordinal)))
                continue;

            policy.Evaluate(action, authority, runtimeProfile.Budget, result);

            if (result.RuleResults.Any(rule =>
                    !rule.IsCompliant && rule.Severity >= runtimeProfile.DenyThreshold))
                break;
        }

        return result;
    }
}
