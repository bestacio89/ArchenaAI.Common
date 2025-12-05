using System.Text.Json.Serialization;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models
{
    /// <summary>
    /// Defines model-specific runtime parameters for LLM requests.
    /// </summary>
    public sealed class LLMRequestParameters
    {
        /// <summary>
        /// Gets or sets the temperature (0 = deterministic, 1 = highly creative).
        /// </summary>
        [JsonPropertyName("temperature")]
        public float? Temperature { get; set; }

        /// <summary>
        /// Gets or sets the nucleus sampling (Top-P) parameter.
        /// </summary>
        [JsonPropertyName("top_p")]
        public float? TopP { get; set; }

        /// <summary>
        /// Gets or sets the maximum number of tokens to generate.
        /// </summary>
        [JsonPropertyName("max_tokens")]
        public int? MaxTokens { get; set; }

        /// <summary>
        /// Gets or sets whether to stream partial completions (for interactive scenarios).
        /// </summary>
        [JsonPropertyName("stream")]
        public bool Stream { get; set; } = false;
    }
}
