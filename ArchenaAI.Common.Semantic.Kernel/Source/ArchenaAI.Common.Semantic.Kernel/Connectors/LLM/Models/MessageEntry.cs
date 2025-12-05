namespace ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models
{
    /// <summary>
    /// Represents a message entry in a conversation (user, assistant, or system).
    /// </summary>
    public sealed class MessageEntry
    {
        /// <summary>
        /// Gets or sets the role of the message (e.g., "system", "user", "assistant").
        /// </summary>
        public string Role { get; set; } = "user";

        /// <summary>
        /// Gets or sets the text content of the message.
        /// </summary>
        public string Content { get; set; } = string.Empty;
    }
}
