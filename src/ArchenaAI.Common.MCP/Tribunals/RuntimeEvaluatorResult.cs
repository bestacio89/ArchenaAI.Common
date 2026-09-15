using Aegis.Shared.Architecture.Models;
using ArchenaAI.Common.MCP.Abstractions;

namespace ArchenaAI.Common.MCP.Tribunals
{
    public static class RuntimeEvaluatorResult
    {
        public static ArchitectureEvaluatorResult ForAction(IRuntimeActionDescriptor action)
            => new(
                source: "MCP.RuntimeTribunal",
                target: action.ActionId,
                domain: "Runtime")
            {
                Category = "Execution",
                Layer = action.OriginLayer
            };
    }
}