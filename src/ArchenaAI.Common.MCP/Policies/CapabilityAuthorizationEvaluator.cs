using ArchenaAI.Common.MCP.Abstractions;
using ArchenaAI.Common.MCP.Models;

namespace ArchenaAI.Common.MCP.Policies;

/// <summary>
/// Checks whether the presented authority grants the required capability
/// and whether that authority has expired. Revocation is enforced by the
/// authority-validation boundary when a revocation store is configured.
/// </summary>
public sealed class CapabilityAuthorizationEvaluator : IPolicyEvaluator
{
    public string RuleId => "AEG-SEC-AUTH001";
    public GovernanceSeverity Severity => GovernanceSeverity.Critical;

    public void Evaluate(
        IRuntimeActionDescriptor action,
        IAuthorityToken authority,
        IExecutionBudget budget,
        GovernanceEvaluationResult result)
    {
        var now = DateTimeOffset.UtcNow;
        var hasGrant = authority.HasCapability(action.CapabilityId);
        var notExpired = authority.ExpiresAt > now;
        var compliant = hasGrant && notExpired;

        var message = !hasGrant
            ? $"Authority '{authority.TokenId}' does not grant capability '{action.CapabilityId}'."
            : !notExpired
                ? $"Authority '{authority.TokenId}' expired at {authority.ExpiresAt:O}."
                : $"Authority '{authority.TokenId}' grants '{action.CapabilityId}'.";

        result.AddRuleResult(new GovernanceRuleResult
        {
            RuleId = RuleId,
            RuleName = "Capability Authorization",
            Severity = Severity,
            IsCompliant = compliant,
            Message = message,
            ActionId = action.ActionId,
            Category = "Security",
            OriginComponent = action.OriginComponent,
            EvaluatedAt = now
        });
    }
}
