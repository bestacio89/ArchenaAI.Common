using ArchenaAI.Common.Semantic.Kernel.Configuration;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models;
using Franz.Common.Errors;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Net.Http.Json;
using System.Text.Json;
using Volo.Abp;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.LLM
{
    /// <summary>
    /// Provides a connector for communicating with remote or hosted Large Language Model (LLM) APIs.
    /// </summary>
    /// <remarks>
    /// The <see cref="LLMConnector"/> class is responsible for formatting requests, sending them to the
    /// configured endpoint (e.g., OpenAI, Azure OpenAI, Anthropic, etc.), and parsing the structured response.
    /// It is designed to remain provider-agnostic while supporting semantic pipeline integration.
    /// </remarks>
    public sealed class LLMConnector : ILLMConnector
    {
        private readonly HttpClient _http;
        private readonly IKernelOptions _options;
        private readonly ILogger<LLMConnector> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="LLMConnector"/> class.
        /// </summary>
        /// <param name="http">The HTTP client used to send model requests.</param>
        /// <param name="options">The kernel configuration options (model, endpoint, temperature, etc.).</param>
        /// <param name="logger">The logger instance for tracing and observability.</param>
        /// <exception cref="ArgumentNullException">
        /// Thrown when <paramref name="http"/>, <paramref name="options"/>, or <paramref name="logger"/> is null.
        /// </exception>
        public LLMConnector(
            HttpClient http,
            IOptions<IKernelOptions> options,
            ILogger<LLMConnector> logger)
        {
            _http = http ?? throw new ArgumentNullException(nameof(http));
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Sends a language model inference request to the configured LLM endpoint.
        /// </summary>
        /// <param name="request">The structured <see cref="LLMRequest"/> containing the prompt and parameters.</param>
        /// <param name="ct">A <see cref="CancellationToken"/> used to cancel the operation if necessary.</param>
        /// <returns>A <see cref="LLMResponse"/> containing the generated model output and metadata.</returns>
        /// <exception cref="BusinessException">Thrown when the <paramref name="request"/> is null.</exception>
        /// <exception cref="HttpRequestException">Thrown when the HTTP request fails or times out.</exception>
        public async Task<LLMResponse> SendAsync(LLMRequest request, CancellationToken ct = default)
        {
            if (request is null)
                throw new BusinessException("[LLM] The request payload cannot be null.");

            JsonContent payload = null!;
            try
            {
                // 🧠 Build the model payload dynamically
                payload = JsonContent.Create(new
                {
                    model = _options.Model,
                    messages = new[]
                    {
                        new { role = "system", content = request.SystemPrompt ?? "You are a helpful AI assistant." },
                        new { role = "user", content = request.Prompt }
                    },
                    temperature = _options.Temperature,
                    max_tokens = _options.MaxTokens,
                    top_p = _options.TopP
                });

                // 🌐 Send the request
                using var response = await _http
                    .PostAsync(new Uri(_options.Endpoint), payload, ct)
                    .ConfigureAwait(false);

                var json = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);

                if (!response.IsSuccessStatusCode)
                {
                    _logger.LogWarning("[LLM] Request failed: {Status} {Content}", response.StatusCode, json);
                    return new LLMResponse
                    {
                        Error = $"HTTP {(int)response.StatusCode}: {response.ReasonPhrase}",
                        StructuredOutput = json
                    };
                }

                // 🧩 Parse JSON response (LLM provider-agnostic)
                using var doc = JsonDocument.Parse(json);
                var root = doc.RootElement;

                var llmResponse = new LLMResponse
                {
                    Id = root.TryGetProperty("id", out var idProp) ? idProp.GetString() : null,
                    Model = root.TryGetProperty("model", out var modelProp) ? modelProp.GetString() : _options.Model,
                    Content = root.TryGetProperty("choices", out var choicesProp)
        ? choicesProp[0].GetProperty("message").GetProperty("content").GetString()
        : null,
                    TotalTokens = root.TryGetProperty("usage", out var usageProp) &&
                  usageProp.TryGetProperty("total_tokens", out var totalTokens)
        ? totalTokens.GetInt32()
        : null,
                    PromptTokens = usageProp.TryGetProperty("prompt_tokens", out var promptTokens)
        ? promptTokens.GetInt32()
        : null,
                    CompletionTokens = usageProp.TryGetProperty("completion_tokens", out var completionTokens)
        ? completionTokens.GetInt32()
        : null
                };

                // Safely populate metadata instead of reassigning it
                llmResponse.Metadata["provider"] = _options.Provider;
                llmResponse.Metadata["endpoint"] = _options.Endpoint;
                llmResponse.Metadata["timestamp"] = DateTime.UtcNow;

                _logger.LogInformation(
                    "[LLM] {Model} | Prompt: {Prompt} | Tokens: {Tokens}",
                    llmResponse.Model,
                    request.Prompt,
                    llmResponse.TotalTokens ?? 0);

                return llmResponse;
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex, "[LLM] Network or protocol error contacting endpoint {Endpoint}", _options.Endpoint);
                throw;
            }
            catch (JsonException ex)
            {
                _logger.LogError(ex, "[LLM] Failed to parse JSON response from {Endpoint}", _options.Endpoint);
                throw;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LLM] Unexpected error during model execution");
                throw;
            }
            finally
            {
                payload?.Dispose();
            }
        }
    }
}
