namespace ArchenaAI.Common.Semantic.Kernel.Configuration
{
    /// <summary>
    /// Configuration contract for the Semantic Kernel and its underlying LLM connectors.
    /// </summary>
    public interface IKernelOptions
    {
        /// <summary>
        /// The logical name of the provider (e.g., "openai", "azure", "anthropic").
        /// </summary>
        string Provider { get; }

        /// <summary>
        /// The REST endpoint for the LLM API.
        /// </summary>
        string Endpoint { get; }

        /// <summary>
        /// The model identifier to use (e.g., "gpt-4o", "gpt-4-turbo", "claude-3").
        /// </summary>
        string Model { get; }

        /// <summary>
        /// The access key or token used for authentication.
        /// </summary>
        string ApiKey { get; }

        /// <summary>
        /// The optional region or deployment name for cloud-hosted providers.
        /// </summary>
        string? Region { get; }

        /// <summary>
        /// The temperature for creative variability in responses (0 = deterministic, 1 = highly creative).
        /// </summary>
        float Temperature { get; }

        /// <summary>
        /// The maximum number of tokens allowed for a single model response.
        /// </summary>
        int MaxTokens { get; }

        /// <summary>
        /// The top-p (nucleus sampling) parameter used for probabilistic token sampling.
        /// </summary>
        float TopP { get; }

        /// <summary>
        /// Indicates whether semantic memory features are enabled (vector recall, context retention, etc.).
        /// </summary>
        bool EnableMemory { get; }

        /// <summary>
        /// Indicates whether telemetry and tracing are enabled.
        /// </summary>
        bool EnableTelemetry { get; }
    }
}
