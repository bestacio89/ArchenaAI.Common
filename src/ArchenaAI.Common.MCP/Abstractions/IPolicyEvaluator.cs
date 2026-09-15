using Aegis.Shared.Architecture.Enums;
using Aegis.Shared.Architecture.Models;

namespace ArchenaAI.Common.MCP.Abstractions
{
    public interface IPolicyEvaluator
    {
        string RuleId { get; }
        ArchitectureRuleSeverity Severity { get; }

        void Evaluate(
            IRuntimeActionDescriptor action,
            IAuthorityToken authority,
            IExecutionBudget budget,
            ArchitectureEvaluatorResult result);
    }
}