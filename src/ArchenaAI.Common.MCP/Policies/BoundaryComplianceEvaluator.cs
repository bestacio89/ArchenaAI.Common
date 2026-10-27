using ArchenaAI.Common.MCP.Abstractions;
using ArchenaAI.Common.MCP.Models;
using System.Linq;

namespace ArchenaAI.Common.MCP.Policies;

/// <summary>
/// Ensures the action stays within the structural constraints of its grant.
/// A missing grant is left to CapabilityAuthorizationEvaluator to report.
/// </summary>
public sealed class BoundaryComplianceEvaluator : IPolicyEvaluator
{
    public string RuleId => "AEG-DEP-BOUND001";
    public GovernanceSeverity Severity => GovernanceSeverity.High;

    public void Evaluate(
        IRuntimeActionDescriptor action,
        IAuthorityToken authority,
        IExecutionBudget budget,
        GovernanceEvaluationResult result)
    {
        var grant = authority.Capabilities.FirstOrDefault(capability => capability.Id == action.CapabilityId);
        if (grant is null)
            return;

        var constraints = grant.Constraints;
        var violations = new List<string>();

        var crossesLayer = !string.Equals(action.OriginLayer, action.TargetLayer, StringComparison.Ordinal);
        if (crossesLayer && !constraints.AllowsCrossDomain)
            violations.Add($"crosses layer '{action.OriginLayer}' -> '{action.TargetLayer}' but capability '{grant.Id}' does not allow cross-domain calls");

        if (constraints.AllowedLayers is { Count: > 0 } allowedLayers
            && !allowedLayers.Contains(action.TargetLayer))
            violations.Add($"target layer '{action.TargetLayer}' is not in the allowed layer set for capability '{grant.Id}'");

        if (constraints.AllowedDomains is { Count: > 0 } allowedDomains
            && !action.ResponsibilityDomains.Any(allowedDomains.Contains))
            violations.Add($"none of the action's responsibility domains are permitted for capability '{grant.Id}'");

        var compliant = violations.Count == 0;
        result.AddRuleResult(new GovernanceRuleResult
        {
            RuleId = RuleId,
            RuleName = "Boundary Compliance",
            Severity = Severity,
            IsCompliant = compliant,
            Message = compliant
                ? $"Action '{action.ActionId}' stays within the boundaries granted by '{grant.Id}'."
                : string.Join("; ", violations),
            ActionId = action.ActionId,
            Category = "Dependency",
            OriginComponent = action.OriginComponent,
            EvaluatedAt = DateTimeOffset.UtcNow
        });
    }
}
