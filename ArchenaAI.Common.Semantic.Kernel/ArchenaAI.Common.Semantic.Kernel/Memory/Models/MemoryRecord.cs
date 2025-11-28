using System.Text.Json.Serialization;

namespace ArchenaAI.Common.Semantic.Kernel.Memory.Models
{
    /// <summary>
    /// Represents a single piece of information stored in semantic memory.
    /// Each record can contain text, embeddings, metadata, and temporal information.
    /// </summary>
    public sealed class MemoryRecord
    {
        private float[]? _embedding;

        /// <summary>
        /// Gets or sets the unique identifier of the memory record.
        /// </summary>
        public string Id { get; set; } = Guid.NewGuid().ToString();

        /// <summary>
        /// Gets or sets the logical category or namespace of this record
        /// (e.g. "user_profile", "conversation", "knowledge").
        /// </summary>
        public string Category { get; set; } = "default";

        /// <summary>
        /// Gets or sets the raw content or textual data stored in this memory entry.
        /// </summary>
        public string Content { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the optional embedding vector representing
        /// the semantic meaning of the content.
        /// </summary>
        /// <remarks>
        /// Returns a defensive copy to prevent mutation of the internal array.
        /// </remarks>
        public IReadOnlyList<float>? Embedding
        {
            get => _embedding is null ? null : Array.AsReadOnly(_embedding);
            set => _embedding = value?.ToArray();
        }

        /// <summary>
        /// Gets the key-value metadata associated with this record.
        /// Example: "source", "confidence", "tags", "author", etc.
        /// </summary>
        public IDictionary<string, object?> Metadata { get; } = new Dictionary<string, object?>();

        /// <summary>
        /// Gets or sets the UTC timestamp of when the record was created or last updated.
        /// </summary>
        public DateTimeOffset Timestamp { get; set; } = DateTimeOffset.UtcNow;

        /// <summary>
        /// Gets or sets the version of this record (useful for distributed stores or replay logs).
        /// </summary>
        public string Version { get; set; } = "1.0";

        /// <summary>
        /// Indicates whether this record is marked as active or archived.
        /// </summary>
        public bool IsActive { get; set; } = true;

        /// <summary>
        /// Gets or sets the similarity score (used for ranking search results).
        /// </summary>
        [JsonIgnore]
        public double? Similarity { get; set; }

        /// <summary>
        /// Adds or updates a metadata entry for this record.
        /// </summary>
        /// <param name="key">The metadata key.</param>
        /// <param name="value">The metadata value.</param>
        /// <returns>The same <see cref="MemoryRecord"/> instance for fluent chaining.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is null or empty.</exception>
        public MemoryRecord AddMetadata(string key, object? value)
        {
            if (string.IsNullOrWhiteSpace(key))
                throw new ArgumentNullException(nameof(key));

            Metadata[key] = value;
            return this;
        }

        /// <summary>
        /// Provides a short description for logging and debugging.
        /// </summary>
        public override string ToString() =>
            $"[{Category}] {Content[..Math.Min(Content.Length, 80)]}... (Id={Id}, Active={IsActive})";
    }
}
