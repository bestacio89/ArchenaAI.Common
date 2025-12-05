using System.Text.Json;

namespace ArchenaAI.Common.Semantic.Kernel.Messaging
{
    /// <summary>
    /// Provides serialization and deserialization for semantic messages exchanged across Kafka.
    /// </summary>
    public sealed class SemanticMessageSerializer : ISemanticMessageSerializer
    {
        private static readonly JsonSerializerOptions _options = new()
        {
            WriteIndented = false,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        /// <inheritdoc/>
        public string Serialize(SemanticMessage message) =>
            JsonSerializer.Serialize(message, _options);

        /// <inheritdoc/>
        public SemanticMessage? Deserialize(string json) =>
            JsonSerializer.Deserialize<SemanticMessage>(json, _options);
    }

    /// <summary>
    /// Defines the contract for serializing semantic messages.
    /// </summary>
    public interface ISemanticMessageSerializer
    {
        string Serialize(SemanticMessage message);
        SemanticMessage? Deserialize(string json);
    }
}
