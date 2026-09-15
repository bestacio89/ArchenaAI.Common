using Aegis.Shared.Architecture.Models;
using ArchenaAI.Common.MCP.Abstractions;
using ArchenaAI.Common.MCP.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ArchenaAI.Common.MCP.Tribunals
{
    /// <summary>
    /// Runs every registered IPolicyEvaluator whose RuleId is enabled by
    /// the McpRuntimeProfile passed to Evaluate. One instance serves every
    /// profile — the profile is what scopes which evaluators actually
    /// fire, so per-profile subclasses would just duplicate that gating.
    /// </summary>
    public sealed class RuntimeTribunal
    {
        private readonly IReadOnlyList<IPolicyEvaluator> _policies;

        public RuntimeTribunal(IEnumerable<IPolicyEvaluator> policies)
        {
            _policies = policies?.ToArray()
                ?? throw new ArgumentNullException(nameof(policies));
        }

        public ArchitectureEvaluatorResult Evaluate(
            IRuntimeActionDescriptor action,
            IAuthorityToken authority,
            McpRuntimeProfile runtimeProfile)
        {
            var result = RuntimeEvaluatorResult.ForAction(action);
            result.Category = runtimeProfile.Name;

            foreach (var policy in _policies)
            {
                if (!runtimeProfile.EnabledRuleIds.Any(id =>
                        id.EndsWith("*")
                            ? policy.RuleId.StartsWith(id.TrimEnd('*'))
                            : policy.RuleId == id))
                    continue;

                policy.Evaluate(action, authority, runtimeProfile.Budget, result);

                // Constitutional stop: only an actual violation at/above the
                // profile's deny threshold halts evaluation.
                if (result.RuleResults.Any(r =>
                        !r.IsCompliant && r.Severity >= runtimeProfile.DenyThreshold))
                {
                    break;
                }
            }

            return result;
        }
    }
}