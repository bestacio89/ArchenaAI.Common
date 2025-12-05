using System.Collections.Generic;

namespace ArchenaAI.Common.Semantic.Kernel.Actions
{
    public interface IActionContext
    {
        string Input { get; }

        IDictionary<string, object?> Arguments { get; }

        string CorrelationId { get; }

        string? CallingAgent { get; }
    }
}
