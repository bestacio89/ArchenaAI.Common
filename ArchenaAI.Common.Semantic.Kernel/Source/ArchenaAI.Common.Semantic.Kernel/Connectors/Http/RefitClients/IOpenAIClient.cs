using Refit;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Http.RefitClients
{
    /// <summary>
    /// Defines a Refit client abstraction for communicating with the
    /// OpenAI or Azure OpenAI REST API.
    /// </summary>
    /// <remarks>
    /// This client is used internally by <see cref="LLM.LLMConnector"/> to send
    /// chat completion or embedding requests to a remote OpenAI-compatible endpoint.
    /// </remarks>
    public interface IOpenAIClient
    {
        /// <summary>
        /// Sends a chat completion request to the OpenAI API.
        /// </summary>
        /// <param name="request">
        /// The payload containing model configuration, messages, and generation parameters.
        /// </param>
        /// <param name="ct">
        /// A <see cref="CancellationToken"/> to cancel the operation if needed.
        /// </param>
        /// <returns>
        /// The raw <see cref="HttpResponseMessage"/> returned by the OpenAI API.
        /// </returns>
        [Post("/v1/chat/completions")]
        Task<HttpResponseMessage> CreateChatCompletionAsync(
            [Body] object request,
            CancellationToken ct = default);

        /// <summary>
        /// Requests an embedding vector from the OpenAI API for a given text input.
        /// </summary>
        /// <param name="request">The payload containing the input text and model name.</param>
        /// <param name="ct">A <see cref="CancellationToken"/> to cancel the operation if needed.</param>
        /// <returns>
        /// The <see cref="HttpResponseMessage"/> containing the embedding response.
        /// </returns>
        [Post("/v1/embeddings")]
        Task<HttpResponseMessage> CreateEmbeddingAsync(
            [Body] object request,
            CancellationToken ct = default);

        /// <summary>
        /// Retrieves metadata about the currently available models from the OpenAI API.
        /// </summary>
        /// <param name="ct">A <see cref="CancellationToken"/> to cancel the operation if needed.</param>
        /// <returns>
        /// A <see cref="HttpResponseMessage"/> containing model metadata as JSON.
        /// </returns>
        [Get("/v1/models")]
        Task<HttpResponseMessage> ListModelsAsync(CancellationToken ct = default);
    }
}
