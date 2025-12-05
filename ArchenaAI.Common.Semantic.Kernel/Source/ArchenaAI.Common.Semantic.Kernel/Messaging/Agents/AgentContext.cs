using System;
using System.Collections.Generic;

namespace ArchenaAI.Common.Semantic.Kernel.Messaging.Agents
{
    public sealed class AgentContext
    {
        public string AgentName { get; init; }
        public string CorrelationId { get; init; }
        public string? Caller { get; init; }
        public Dictionary<string, object?> Metadata { get; init; }

        public AgentContext(
            string agentName,
            string correlationId,
            string? caller,
            Dictionary<string, object?> metadata)
        {
            AgentName = agentName;
            CorrelationId = correlationId;
            Caller = caller;
            Metadata = metadata;
        }
    }
}
