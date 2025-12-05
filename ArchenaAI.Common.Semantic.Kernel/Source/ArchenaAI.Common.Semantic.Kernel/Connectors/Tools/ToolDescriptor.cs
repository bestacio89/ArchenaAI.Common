using System;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Tools
{
    /// <summary>
    /// Represents metadata for a registered tool executor.
    /// </summary>
    public sealed class ToolDescriptor
    {
        /// <summary>
        /// Gets the name of the tool (e.g., "terraform", "bash").
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Gets the tool’s purpose or usage description.
        /// </summary>
        public string Description { get; }

        /// <summary>
        /// Gets the registration timestamp (UTC).
        /// </summary>
        public DateTimeOffset RegisteredAt { get; }

        public ToolDescriptor(string name, string description)
        {
            Name = name;
            Description = description;
            RegisteredAt = DateTimeOffset.UtcNow;
        }

        public override string ToString() => $"{Name} — {Description} (Registered: {RegisteredAt:u})";
    }
}
