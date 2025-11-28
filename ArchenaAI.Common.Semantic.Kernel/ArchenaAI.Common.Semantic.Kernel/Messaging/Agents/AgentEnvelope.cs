using System;
using System.Collections.Generic;

namespace ArchenaAI.Common.Semantic.Kernel.Messaging.Agents
{
    /// <summary>
    /// Standard message envelope used by all ArchenaAI agents.
    /// Contains routing, correlation, payload, and metadata information.
    /// </summary>
    public sealed class AgentEnvelope
    {
        /// <summary>
        /// Strongly-typed message classification.
        /// Used for routing and for agent capability checks.
        /// </summary>
        public AgentMessageType Type { get; set; } = AgentMessageType.Request;

        /// <summary>
        /// Target agent name (for direct routing).
        /// Required when routing is not capability-based.
        /// </summary>
        public string Agent { get; set; } = string.Empty;

        /// <summary>
        /// Unique correlation identifier for request/response tracking.
        /// </summary>
        public string CorrelationId { get; set; } = Guid.NewGuid().ToString();

        /// <summary>
        /// Kafka topic to send responses to.
        /// If null, agent default topic is used.
        /// </summary>
        public string? ReplyTo { get; set; }

        /// <summary>
        /// Raw payload. Typically natural language instructions,
        /// serialized tool outputs, or structured memory data.
        /// </summary>
        public string Payload { get; set; } = string.Empty;

        /// <summary>
        /// Additional routing and semantic metadata.
        /// Pluggable, case-insensitive dictionary.
        /// </summary>
        public Dictionary<string, object?> Metadata { get; set; }
            = new(StringComparer.OrdinalIgnoreCase);
    }
}
