using ArchenaAI.Common.MCP.Models;

namespace ArchenaAI.Common.MCP.Abstractions;

public interface IPolicyEvaluator
{
    string RuleId { get; }
    GovernanceSeverity Severity { get; }

    void Evaluate(
        IRuntimeActionDescriptor action,
        IAuthorityToken authority,
        IExecutionBudget budget,
        GovernanceEvaluationResult result);
}
