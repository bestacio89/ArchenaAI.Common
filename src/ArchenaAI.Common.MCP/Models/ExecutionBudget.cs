using ArchenaAI.Common.MCP.Abstractions;
using System;
using System.Collections.Generic;
using System.Text;

namespace ArchenaAI.Common.MCP.Models
{
    public sealed partial record ExecutionBudget : IExecutionBudget
    {
        public int MaxReasoningSteps { get; init; }
        public int MaxToolCalls { get; init; }
        public TimeSpan MaxExecutionTime { get; init; }
        public int MaxCostUnits { get; init; }
    }

}
