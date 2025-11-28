namespace ArchenaAI.Common.Semantic.Kernel.Messaging
{
    /// <summary>
    /// Represents a standardized message envelope used by semantic agents
    /// to exchange structured information over Kafka.
    /// </summary>
    public sealed record SemanticMessage
    {
        /// <summary>
        /// Gets or sets the logical category or domain of the event.
        /// Example: <c>memory.update</c>, <c>skill.invoke</c>, <c>reasoning.feedback</c>.
        /// </summary>
        public string Type { get; init; } = string.Empty;

        /// <summary>
        /// Gets or sets the serialized message content or payload.
        /// </summary>
        public string Content { get; init; } = string.Empty;

        /// <summary>
        /// Gets or sets the timestamp when the message was produced.
        /// </summary>
        public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

        /// <summary>
        /// Gets or sets the ID of the originating agent or service.
        /// </summary>
        public string Source { get; init; } = "semantic.kernel";

        /// <summary>
        /// Gets or sets optional metadata, such as correlation IDs or embeddings.
        /// </summary>
        public IDictionary<string, object?> Metadata { get; init; } = new Dictionary<string, object?>();
    }
}
