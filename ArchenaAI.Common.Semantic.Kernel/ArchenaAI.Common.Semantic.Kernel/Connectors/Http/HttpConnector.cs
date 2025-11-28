using Microsoft.Extensions.Logging;
using System.Net.Http.Json;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Http
{
    /// <summary>
    /// Default implementation of <see cref="IHttpConnector"/> that wraps <see cref="HttpClient"/>
    /// for performing safe and testable HTTP communications.
    /// </summary>
    /// <remarks>
    /// This class provides strongly-typed methods for performing <c>GET</c> and <c>POST</c> requests,
    /// and ensures compliance with analyzer and reliability rules (disposal, argument validation, etc.).
    /// </remarks>
    public sealed class HttpConnector : IHttpConnector
    {
        private readonly HttpClient _http;
        private readonly ILogger<HttpConnector> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="HttpConnector"/> class.
        /// </summary>
        /// <param name="http">The <see cref="HttpClient"/> used for sending HTTP requests.</param>
        /// <param name="logger">The logger instance for observability and diagnostics.</param>
        /// <exception cref="ArgumentNullException">
        /// Thrown when <paramref name="http"/> or <paramref name="logger"/> is <see langword="null"/>.
        /// </exception>
        public HttpConnector(HttpClient http, ILogger<HttpConnector> logger)
        {
            _http = http ?? throw new ArgumentNullException(nameof(http));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <inheritdoc/>
        public async Task<HttpResponseMessage> PostAsync<T>(Uri uri, T payload, CancellationToken ct)
        {
            if (uri is null)
                throw new ArgumentNullException(nameof(uri));
            if (payload is null)
                throw new ArgumentNullException(nameof(payload));

            _logger.LogDebug("[HTTP] POST {Uri} | Payload Type: {Type}", uri, typeof(T).Name);

            try
            {
                using var content = JsonContent.Create(payload);
                var response = await _http.PostAsync(uri, content, ct).ConfigureAwait(false);
                _logger.LogInformation("[HTTP] POST {Uri} -> {StatusCode}", uri, response.StatusCode);
                return response;
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex, "[HTTP] Network error calling {Uri}", uri);
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<HttpResponseMessage> GetAsync(Uri uri, CancellationToken ct)
        {
            if (uri is null)
                throw new ArgumentNullException(nameof(uri));

            _logger.LogDebug("[HTTP] GET {Uri}", uri);

            try
            {
                var response = await _http.GetAsync(uri, ct).ConfigureAwait(false);
                _logger.LogInformation("[HTTP] GET {Uri} -> {StatusCode}", uri, response.StatusCode);
                return response;
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex, "[HTTP] Network error calling {Uri}", uri);
                throw;
            }
        }
    }
}
