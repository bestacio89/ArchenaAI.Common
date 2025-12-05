using System.Text.Json.Serialization;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models
{
    /// <summary>
    /// Represents a standardized request to a Large Language Model (LLM).
    /// </summary>
    public sealed class LLMRequest
    {
        /// <summary>
        /// Gets or sets the main prompt or instruction to send to the model.
        /// </summary>
        public string Prompt { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the optional system message that defines the assistant's behavior or persona.
        /// </summary>
        public string? SystemPrompt { get; set; }

        /// <summary>
        /// Gets or sets any additional context or conversation history (user and assistant turns).
        /// </summary>
        public IList<MessageEntry> Context { get;  } = new List<MessageEntry>();

        /// <summary>
        /// Gets or sets optional parameters that may override the default model configuration for this call.
        /// </summary>
        public LLMRequestParameters? Parameters { get; set; }

        /// <summary>
        /// Gets or sets arbitrary metadata for correlation, tracing, or downstream consumers.
        /// </summary>
        public IDictionary<string, object?> Metadata { get; } = new Dictionary<string, object?>();

        /// <summary>
        /// Returns a human-readable representation of the request.
        /// </summary>
        public override string ToString() => $"{Prompt[..Math.Min(Prompt.Length, 80)]}...";
    }
}
