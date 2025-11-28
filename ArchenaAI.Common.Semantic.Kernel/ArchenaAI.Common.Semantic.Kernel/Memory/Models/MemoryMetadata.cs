using System;
using System.Collections.Generic;

namespace ArchenaAI.Common.Semantic.Kernel.Memory.Models
{
    /// <summary>
    /// Represents structured metadata associated with a memory record.
    /// Provides contextual information such as origin, confidence, and tags.
    /// </summary>
    public sealed class MemoryMetadata
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="MemoryMetadata"/> class.
        /// </summary>
        public MemoryMetadata()
        {
            Tags = new List<string>();
            Attributes = new Dictionary<string, object?>();
        }

        /// <summary>
        /// Gets or sets the source or origin of the memory entry (e.g., "conversation", "document", "system").
        /// </summary>
        public string Source { get; set; } = "unknown";

        /// <summary>
        /// Gets or sets the author or agent responsible for this memory entry.
        /// </summary>
        public string Author { get; set; } = "system";

        /// <summary>
        /// Gets or sets a confidence score (0.0–1.0) indicating the reliability or certainty of this memory.
        /// </summary>
        public double Confidence { get; set; } = 1.0;

        /// <summary>
        /// Gets or sets the timestamp when this metadata was created or last updated.
        /// </summary>
        public DateTimeOffset Timestamp { get; set; } = DateTimeOffset.UtcNow;

        /// <summary>
        /// Gets the collection of tags categorizing this memory (e.g., "summary", "fact", "context").
        /// </summary>
        public IList<string> Tags { get; }

        /// <summary>
        /// Gets the collection of additional attributes or extended key-value pairs.
        /// </summary>
        public IDictionary<string, object?> Attributes { get; }

        /// <summary>
        /// Returns a human-readable summary of the metadata contents.
        /// </summary>
        /// <returns>A formatted string describing the metadata fields.</returns>
        public override string ToString() =>
            $"Source={Source}, Author={Author}, Confidence={Confidence:F2}, Tags=[{string.Join(", ", Tags)}]";
    }
}
