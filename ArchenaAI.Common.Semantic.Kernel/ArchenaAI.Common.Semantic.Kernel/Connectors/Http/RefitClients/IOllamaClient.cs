using Refit;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Http.RefitClients
{
    /// <summary>
    /// Defines a Refit client for communicating with a locally hosted
    /// or containerized Ollama model server (or any compatible LLM API).
    /// </summary>
    /// <remarks>
    /// This client is typically used by the <see cref="Locals.LocalModelConnector"/>
    /// to send inference and model management requests to the local runtime.
    /// </remarks>
    public interface IOllamaClient
    {
        /// <summary>
        /// Sends an inference request to the Ollama API to generate a model response.
        /// </summary>
        /// <param name="request">The payload containing model, prompt, and parameters.</param>
        /// <param name="ct">A <see cref="CancellationToken"/> to cancel the operation if needed.</param>
        /// <returns>
        /// The HTTP response returned by the Ollama service.
        /// </returns>
        [Post("/api/generate")]
        Task<HttpResponseMessage> GenerateAsync(
            [Body] object request,
            CancellationToken ct = default);

        /// <summary>
        /// Retrieves a list of locally available models from the Ollama runtime.
        /// </summary>
        /// <param name="ct">A <see cref="CancellationToken"/> to cancel the operation if needed.</param>
        /// <returns>
        /// The HTTP response containing the list of models as JSON.
        /// </returns>
        [Get("/api/tags")]
        Task<HttpResponseMessage> ListModelsAsync(CancellationToken ct = default);

        /// <summary>
        /// Pulls a model from a remote registry into the local Ollama store.
        /// </summary>
        /// <param name="modelName">The name of the model to pull (e.g., <c>llama2</c>).</param>
        /// <param name="ct">A <see cref="CancellationToken"/> to cancel the operation if needed.</param>
        /// <returns>
        /// The HTTP response from the Ollama service.
        /// </returns>
        [Post("/api/pull")]
        Task<HttpResponseMessage> PullModelAsync(
            [Body] object modelName,
            CancellationToken ct = default);
    }
}
