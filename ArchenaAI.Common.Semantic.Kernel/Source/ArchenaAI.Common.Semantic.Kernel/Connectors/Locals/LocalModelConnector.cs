using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Net.Http.Json;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Locals
{
    /// <summary>
    /// Provides connectivity to locally hosted or containerized language models such as
    /// Ollama, LM Studio, or other custom inference endpoints.
    /// </summary>
    /// <remarks>
    /// This connector acts as a lightweight adapter for sending prompts and retrieving
    /// responses from local model runtimes. It is primarily designed for fast, offline,
    /// or self-hosted LLM inference.
    /// </remarks>
    public sealed class LocalModelConnector : ILocalModelConnector
    {
        private readonly HttpClient _http;
        private readonly LocalModelOptions _options;
        private readonly ILogger<LocalModelConnector> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="LocalModelConnector"/> class.
        /// </summary>
        /// <param name="http">The HTTP client used to communicate with the local model endpoint.</param>
        /// <param name="options">Configuration options specifying runtime, model path, and endpoint.</param>
        /// <param name="logger">The logging instance used for observability and error tracking.</param>
        /// <exception cref="ArgumentNullException">
        /// Thrown when <paramref name="http"/>, <paramref name="options"/>, or <paramref name="logger"/> is null.
        /// </exception>
        public LocalModelConnector(
            HttpClient http,
            IOptions<LocalModelOptions> options,
            ILogger<LocalModelConnector> logger)
        {
            _http = http ?? throw new ArgumentNullException(nameof(http));
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Sends an inference request to the local model and returns the generated response.
        /// </summary>
        /// <param name="request">The language model request containing the prompt and inference parameters.</param>
        /// <param name="ct">A cancellation token that can be used to cancel the operation.</param>
        /// <returns>A <see cref="LLMResponse"/> containing the generated text or model output.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="request"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the request prompt is null or empty.</exception>
        /// <exception cref="HttpRequestException">Thrown when the local endpoint cannot be reached or returns an error.</exception>
        /// <exception cref="Exception">Thrown for any unexpected runtime exception during inference.</exception>
        public async Task<LLMResponse> InferAsync(LLMRequest request, CancellationToken ct)
        {
            ArgumentNullException.ThrowIfNull(request);

            if (string.IsNullOrWhiteSpace(request.Prompt))
                throw new ArgumentException("Prompt cannot be null or empty.", nameof(request));

            _logger.LogDebug("[LocalModel] Sending inference for prompt length: {Length}", request.Prompt.Length);

            var payload = new
            {
                model = _options.ModelPath,
                system_prompt = request.SystemPrompt,
                messages = request.Context?.Select(m => new { role = m.Role, content = m.Content }).ToList(),
                prompt = request.Prompt,
                parameters = new
                {
                    temperature = request.Parameters?.Temperature ?? 0.7f,
                    top_p = request.Parameters?.TopP ?? 1.0f,
                    max_new_tokens = request.Parameters?.MaxTokens ?? 512,
                    stream = request.Parameters?.Stream ?? false
                }
            };

            try
            {
                var endpoint = $"{_options.Endpoint.TrimEnd('/')}/generate";
                var response = await _http.PostAsJsonAsync(endpoint, payload, ct).ConfigureAwait(false);
                response.EnsureSuccessStatusCode();

                var parsed = await response.Content.ReadFromJsonAsync<LLMResponse>(cancellationToken: ct)
                    .ConfigureAwait(false)
                    ?? new LLMResponse { Content = "[empty response]" };

                _logger.LogInformation(
                    "[LocalModel] Response received successfully: {Preview}",
                    parsed.Content?.Length > 100
                        ? parsed.Content[..100] + "..."
                        : parsed.Content);

                return parsed;
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex, "[LocalModel] Network error calling local endpoint {Endpoint}", _options.Endpoint);
                throw;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LocalModel] Unexpected error while processing local inference");
                throw;
            }
        }
    }
}
