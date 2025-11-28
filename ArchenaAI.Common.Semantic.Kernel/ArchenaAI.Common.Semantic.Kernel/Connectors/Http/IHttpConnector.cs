using System.Net.Http;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Http
{
    /// <summary>
    /// Defines an abstraction for performing HTTP communication with external APIs
    /// in a provider-agnostic and testable way.
    /// </summary>
    /// <remarks>
    /// This interface is used by connector components such as LLM or embedding providers
    /// to standardize outbound HTTP operations while allowing easy mocking in unit tests.
    /// </remarks>
    public interface IHttpConnector
    {
        /// <summary>
        /// Sends an HTTP <c>POST</c> request to the specified endpoint with a typed payload.
        /// </summary>
        /// <typeparam name="T">The type of the payload to send in the request body.</typeparam>
        /// <param name="uri">The target endpoint as a <see cref="Uri"/> instance.</param>
        /// <param name="payload">The strongly-typed request body to serialize and send.</param>
        /// <param name="ct">A <see cref="CancellationToken"/> to observe while waiting for the task to complete.</param>
        /// <returns>
        /// A <see cref="HttpResponseMessage"/> representing the response from the remote endpoint.
        /// </returns>
        /// <exception cref="ArgumentNullException">
        /// Thrown when <paramref name="uri"/> or <paramref name="payload"/> is <see langword="null"/>.
        /// </exception>
        /// <exception cref="HttpRequestException">
        /// Thrown when the request fails due to network or protocol errors.
        /// </exception>
        Task<HttpResponseMessage> PostAsync<T>(Uri uri, T payload, CancellationToken ct);

        /// <summary>
        /// Sends an HTTP <c>GET</c> request to the specified endpoint.
        /// </summary>
        /// <param name="uri">The target endpoint as a <see cref="Uri"/> instance.</param>
        /// <param name="ct">A <see cref="CancellationToken"/> to observe while waiting for the task to complete.</param>
        /// <returns>
        /// A <see cref="HttpResponseMessage"/> containing the server’s response message.
        /// </returns>
        /// <exception cref="ArgumentNullException">
        /// Thrown when <paramref name="uri"/> is <see langword="null"/>.
        /// </exception>
        /// <exception cref="HttpRequestException">
        /// Thrown when the request fails due to network or protocol errors.
        /// </exception>
        Task<HttpResponseMessage> GetAsync(Uri uri, CancellationToken ct);
    }
}
