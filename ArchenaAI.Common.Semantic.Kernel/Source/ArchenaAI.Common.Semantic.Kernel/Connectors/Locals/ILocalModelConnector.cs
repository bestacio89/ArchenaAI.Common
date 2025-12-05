using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Locals
{
    /// <summary>
    /// Defines a contract for connectors that interact with locally hosted or containerized language models.
    /// </summary>
    /// <remarks>
    /// Implementations of this interface provide the ability to send inference requests to
    /// local model runtimes such as <c>Ollama</c>, <c>LM Studio</c>, or custom Python/C++ inference servers.
    /// This abstraction allows the Semantic Kernel to remain agnostic of specific backend implementations.
    /// </remarks>
    public interface ILocalModelConnector
    {
        /// <summary>
        /// Sends an inference request to the local model and retrieves the generated response.
        /// </summary>
        /// <param name="request">
        /// The <see cref="LLMRequest"/> object containing the prompt, parameters, and conversation context
        /// to be processed by the local model.
        /// </param>
        /// <param name="ct">A <see cref="CancellationToken"/> used to cancel the operation if needed.</param>
        /// <returns>
        /// A task that represents the asynchronous operation.  
        /// The task result contains an <see cref="LLMResponse"/> with the model's generated output.
        /// </returns>
        /// <exception cref="ArgumentNullException">
        /// Thrown when the <paramref name="request"/> is <see langword="null"/>.
        /// </exception>
        /// <exception cref="ArgumentException">
        /// Thrown when the <paramref name="request"/> prompt is <see langword="null"/> or empty.
        /// </exception>
        /// <exception cref="HttpRequestException">
        /// Thrown when the local model endpoint cannot be reached or returns an error response.
        /// </exception>
        Task<LLMResponse> InferAsync(LLMRequest request, CancellationToken ct);
    }
}
