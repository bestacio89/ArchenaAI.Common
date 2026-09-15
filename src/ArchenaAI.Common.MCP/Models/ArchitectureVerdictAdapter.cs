using Aegis.Shared.Architecture.Enums;
using Aegis.Shared.Architecture.Models;
using Aegis.Shared.Architecture.Models.Rules;
using ArchenaAI.Common.MCP.Abstractions;
using System.Collections.Generic;
using System.Linq;

namespace ArchenaAI.Common.MCP.Models
{
    public sealed class ArchitectureVerdictAdapter : IVerdict
    {
        private readonly ArchitectureEvaluatorResult _result;
        private readonly ArchitectureRuleSeverity _denyThreshold;

        public ArchitectureVerdictAdapter(
            ArchitectureEvaluatorResult result,
            ArchitectureRuleSeverity denyThreshold)
        {
            _result = result;
            _denyThreshold = denyThreshold;
        }

        // Only genuine violations count — a passing rule tagged Critical
        // is not a reason to deny anything.
        private IEnumerable<ArchitectureRuleresult> Violations =>
            _result.RuleResults.Where(r => !r.IsCompliant);

        public bool IsAllowed =>
            !Violations.Any(r => r.Severity >= _denyThreshold);

        public ArchitectureRuleSeverity MaxSeverity =>
            Violations.Any()
                ? Violations.Max(r => r.Severity)
                : ArchitectureRuleSeverity.Info;

        public IReadOnlyCollection<string> ViolatedRuleIds =>
            Violations.Select(r => r.RuleId).ToArray();
    }
}