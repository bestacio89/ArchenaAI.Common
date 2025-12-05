using System;
using System.Collections.Generic;

namespace ArchenaAI.Common.Semantic.Kernel.Actions
{
    public sealed class ActionContext : IActionContext
    {
        public string Input { get; }
        public IDictionary<string, object?> Arguments { get; }
        public string CorrelationId { get; }
        public string? CallingAgent { get; }

        public ActionContext(
            string input,
            IDictionary<string, object?> args,
            string? caller,
            string correlationId)
        {
            Input = input;
            Arguments = args;
            CallingAgent = caller;
            CorrelationId = correlationId;
        }
    }
}
