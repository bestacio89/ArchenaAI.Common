using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.LLM
{
    /// <summary>
    /// Defines a contract for connectors that interact with remote or hosted
    /// Large Language Model (LLM) APIs such as OpenAI, Azure OpenAI, Anthropic, or similar providers.
    /// </summary>
    /// <remarks>
    /// Implementations of this interface are responsible for:
    /// <list type="bullet">
    /// <item><description>Formatting and sending model inference requests to an API endpoint.</description></item>
    /// <item><description>Parsing and returning structured responses through <see cref="LLMResponse"/>.</description></item>
    /// <item><description>Providing provider-agnostic communication for the Semantic Kernel pipelines.</description></item>
    /// </list>
    /// </remarks>
    public interface ILLMConnector
    {
        /// <summary>
        /// Sends an inference request to the configured language model endpoint and retrieves the generated response.
        /// </summary>
        /// <param name="request">
        /// The <see cref="LLMRequest"/> containing the prompt, system message, and inference parameters to send to the model.
        /// </param>
        /// <param name="ct">
        /// A <see cref="CancellationToken"/> used to cancel the operation if necessary.
        /// </param>
        /// <returns>
        /// A task that represents the asynchronous operation.  
        /// The task result contains an <see cref="LLMResponse"/> with the model's generated output and metadata.
        /// </returns>
        /// <exception cref="ArgumentNullException">
        /// Thrown when the <paramref name="request"/> is <see langword="null"/>.
        /// </exception>
        /// <exception cref="ArgumentException">
        /// Thrown when the <paramref name="request"/> prompt is <see langword="null"/> or empty.
        /// </exception>
        /// <exception cref="HttpRequestException">
        /// Thrown when the remote endpoint cannot be reached or returns a failed HTTP status code.
        /// </exception>
        Task<LLMResponse> SendAsync(LLMRequest request, CancellationToken ct);
    }
}
