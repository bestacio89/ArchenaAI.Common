using Aegis.Shared.Architecture.Enums;
using Aegis.Shared.Architecture.Models;
using Aegis.Shared.Architecture.Models.Rules;
using ArchenaAI.Common.MCP.Abstractions;
using System;

namespace ArchenaAI.Common.MCP.Policies
{
    /// <summary>
    /// Verifies the presented AuthorityToken actually grants the capability
    /// the action requires, and that the grant hasn't expired.
    /// NOTE: AuthorityToken.Revocable only says a token CAN be revoked, not
    /// that it currently IS — there's no revocation store yet, so an active
    /// revocation isn't checked here. That needs a token store, which is a
    /// separate piece of work.
    /// </summary>
    public sealed class CapabilityAuthorizationEvaluator : IPolicyEvaluator
    {
        public string RuleId => "AEG-SEC-AUTH001";
        public ArchitectureRuleSeverity Severity => ArchitectureRuleSeverity.Critical;

        public void Evaluate(
            IRuntimeActionDescriptor action,
            IAuthorityToken authority,
            IExecutionBudget budget,
            ArchitectureEvaluatorResult result)
        {
            var now = DateTimeOffset.UtcNow;
            bool hasGrant = authority.HasCapability(action.CapabilityId);
            bool notExpired = authority.ExpiresAt > now;
            bool isCompliant = hasGrant && notExpired;

            var message = !hasGrant
                ? $"Authority '{authority.TokenId}' does not grant capability '{action.CapabilityId}'."
                : !notExpired
                    ? $"Authority '{authority.TokenId}' expired at {authority.ExpiresAt:O}."
                    : $"Authority '{authority.TokenId}' grants '{action.CapabilityId}'.";

            result.AddRuleResult(new ArchitectureRuleresult(
                ruleId: RuleId,
                ruleName: "Capability Authorization",
                category: ArchitectureRuleCategory.Security,
                severity: Severity,
                filePath: null,
                @namespace: action.OriginComponent,
                message: message,
                detectedAt: now,
                isCompliant: isCompliant)
            {
                Target = action.ActionId,
                Domain = "Runtime",
                DetectedBy = nameof(CapabilityAuthorizationEvaluator)
            });
        }
    }
}