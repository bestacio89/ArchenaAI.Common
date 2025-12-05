using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace ArchenaAI.Common.Semantic.Kernel.Messaging.Agents
{
    /// <summary>
    /// Responsible for serializing and deserializing agent envelopes used
    /// for distributed reasoning and multi-agent communication.
    /// </summary>
    public static class AgentMessageSerializer
    {
        private static readonly JsonSerializerOptions Options = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = false,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            Converters =
            {
                new JsonStringEnumConverter(JsonNamingPolicy.CamelCase)
            }
        };

        /// <summary>
        /// Serializes an <see cref="AgentEnvelope"/> into a compact JSON string.
        /// </summary>
        public static string Serialize(AgentEnvelope envelope)
        {
            if (envelope == null)
                throw new ArgumentNullException(nameof(envelope));

            return JsonSerializer.Serialize(envelope, Options);
        }

        /// <summary>
        /// Deserializes JSON back into an <see cref="AgentEnvelope"/>.
        /// </summary>
        public static AgentEnvelope Deserialize(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
                throw new ArgumentException("Message cannot be null or empty.", nameof(json));

            var envelope = JsonSerializer.Deserialize<AgentEnvelope>(json, Options);

            if (envelope == null)
                throw new InvalidOperationException("Failed to deserialize agent envelope.");

            return envelope;
        }
    }
}
