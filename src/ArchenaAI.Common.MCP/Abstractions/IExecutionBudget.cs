using System;

namespace ArchenaAI.Common.MCP.Abstractions
{
    public interface IExecutionBudget
    {
        int MaxReasoningSteps { get; }
        int MaxToolCalls { get; }
        TimeSpan MaxExecutionTime { get; }
        int MaxCostUnits { get; }
    }
}