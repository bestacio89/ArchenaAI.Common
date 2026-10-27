using ArchenaAI.Common.MCP.Abstractions;
using ArchenaAI.Common.MCP.Models;

namespace ArchenaAI.Common.MCP.Tribunals;

public static class RuntimeEvaluatorResult
{
    public static GovernanceEvaluationResult ForAction(IRuntimeActionDescriptor action) =>
        new()
        {
            Source = "MCP.RuntimeTribunal",
            Target = action.ActionId,
            Category = "Execution"
        };
}
