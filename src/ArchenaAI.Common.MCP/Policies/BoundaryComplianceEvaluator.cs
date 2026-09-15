using Aegis.Shared.Architecture.Enums;
using Aegis.Shared.Architecture.Models;
using Aegis.Shared.Architecture.Models.Rules;
using ArchenaAI.Common.MCP.Abstractions;
using ArchenaAI.Common.MCP.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ArchenaAI.Common.MCP.Policies
{
    /// <summary>
    /// Verifies a runtime action doesn't cross a domain/layer boundary the
    /// granted capability doesn't allow. Assumes CapabilityAuthorizationEvaluator
    /// already confirmed the grant exists — if it's missing, that's a
    /// different rule's failure, so this one is a no-op.
    /// </summary>
    public sealed class BoundaryComplianceEvaluator : IPolicyEvaluator
    {
        public string RuleId => "AEG-DEP-BOUND001";
        public ArchitectureRuleSeverity Severity => ArchitectureRuleSeverity.High;

        public void Evaluate(
            IRuntimeActionDescriptor action,
            IAuthorityToken authority,
            IExecutionBudget budget,
            ArchitectureEvaluatorResult result)
        {
            var grant = authority.Capabilities
                .FirstOrDefault(c => c.Id == action.CapabilityId);

            if (grant is null)
                return;

            var constraints = grant.Constraints;
            var violations = new List<string>();

            bool crossesLayer = !string.Equals(action.OriginLayer, action.TargetLayer, StringComparison.Ordinal);
            if (crossesLayer && !constraints.AllowsCrossDomain)
                violations.Add($"crosses layer '{action.OriginLayer}' -> '{action.TargetLayer}' but capability '{grant.Id}' does not allow cross-domain calls");

            if (constraints.AllowedLayers is { Count: > 0 } allowedLayers
                && !allowedLayers.Contains(action.TargetLayer))
                violations.Add($"target layer '{action.TargetLayer}' is not in the allowed layer set for capability '{grant.Id}'");

            if (constraints.AllowedDomains is { Count: > 0 } allowedDomains
                && !action.ResponsibilityDomains.Any(d => allowedDomains.Contains(d)))
                violations.Add($"none of the action's responsibility domains are permitted for capability '{grant.Id}'");

            bool isCompliant = violations.Count == 0;
            var message = isCompliant
                ? $"Action '{action.ActionId}' stays within the boundaries granted by '{grant.Id}'."
                : string.Join("; ", violations);

            result.AddRuleResult(new ArchitectureRuleresult(
                ruleId: RuleId,
                ruleName: "Boundary Compliance",
                category: ArchitectureRuleCategory.Dependency,
                severity: Severity,
                filePath: null,
                @namespace: action.OriginComponent,
                message: message,
                detectedAt: DateTimeOffset.UtcNow,
                isCompliant: isCompliant)
            {
                Target = action.ActionId,
                Domain = "Runtime",
                DetectedBy = nameof(BoundaryComplianceEvaluator)
            });
        }
    }
}