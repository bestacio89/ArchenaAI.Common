using ArchenaAI.Common.MCP.Abstractions;
using ArchenaAI.Common.MCP.Models;

namespace ArchenaAI.Common.MCP.Tribunals;

/// <summary>
/// Runs only policy rules enabled by the supplied profile and enforces the
/// profile's capability allow-list before invoking any policy evaluator.
/// </summary>
public sealed class RuntimeTribunal
{
    private const string ProfileCapabilityRuleId = "AEG-SEC-PROFILE001";
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

        var profileGrant = runtimeProfile.AllowedCapabilities
            .FirstOrDefault(capability => string.Equals(
                capability.Id, action.CapabilityId, StringComparison.Ordinal));

        if (profileGrant is null)
        {
            result.AddRuleResult(new GovernanceRuleResult
            {
                RuleId = ProfileCapabilityRuleId,
                RuleName = "Runtime Profile Capability Allow-List",
                Severity = GovernanceSeverity.Critical,
                IsCompliant = false,
                Message = $"Runtime profile '{runtimeProfile.Name}' does not allow capability '{action.CapabilityId}'.",
                ActionId = action.ActionId,
                Category = "Security",
                OriginComponent = action.OriginComponent,
                EvaluatedAt = DateTimeOffset.UtcNow
            });

            return result;
        }

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
