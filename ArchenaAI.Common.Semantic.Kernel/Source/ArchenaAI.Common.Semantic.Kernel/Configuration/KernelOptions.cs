namespace ArchenaAI.Common.Semantic.Kernel.Configuration
{
    /// <summary>
    /// Default implementation of <see cref="IKernelOptions"/> for dependency injection and configuration binding.
    /// </summary>
    public sealed class KernelOptions : IKernelOptions
    {
        /// <summary>
        /// Gets or sets the logical name of the provider (e.g., "openai", "azure", "anthropic").
        /// </summary>
        public string Provider { get; set; } = "openai";

        /// <summary>
        /// Gets or sets the endpoint for the LLM API.
        /// </summary>
        public string Endpoint { get; set; } = "https://api.openai.com/v1/chat/completions";

        /// <summary>
        /// Gets or sets the model identifier (e.g., "gpt-4o", "gpt-4-turbo", "claude-3").
        /// </summary>
        public string Model { get; set; } = "gpt-4o";

        /// <summary>
        /// Gets or sets the authentication key for the LLM API.
        /// </summary>
        public string ApiKey { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the optional region (for Azure or cloud-based deployments).
        /// </summary>
        public string? Region { get; set; }

        /// <summary>
        /// Gets or sets the temperature parameter controlling randomness.
        /// </summary>
        public float Temperature { get; set; } = 0.7f;

        /// <summary>
        /// Gets or sets the maximum number of tokens for responses.
        /// </summary>
        public int MaxTokens { get; set; } = 4096;

        /// <summary>
        /// Gets or sets the nucleus sampling parameter (Top-p).
        /// </summary>
        public float TopP { get; set; } = 1.0f;

        /// <summary>
        /// Gets or sets whether memory features (vector recall, long-term context) are enabled.
        /// </summary>
        public bool EnableMemory { get; set; } =  false;

        /// <summary>
        /// Gets or sets whether telemetry and distributed tracing are enabled.
        /// </summary>
        public bool EnableTelemetry { get; set; } = true;
    }
}
