using ArchenaAI.Common.Semantic.Kernel.Memory.Contracts;
using ArchenaAI.Common.Semantic.Kernel.Memory.Models;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Memory
{
    /// <summary>
    /// Provides vector embeddings for text content using an external model endpoint (LLM or embedding API).
    /// </summary>
    public sealed class VectorMemoryProvider : IVectorMemoryProvider
    {
        private readonly HttpClient _http;
        private readonly ILogger<VectorMemoryProvider> _logger;

        // Temporary in-memory vector store (until replaced by a vector DB)
        private readonly Dictionary<string, MemoryRecord> _store = new();

        /// <summary>
        /// Initializes a new instance of the <see cref="VectorMemoryProvider"/> class.
        /// </summary>
        /// <param name="http">The HTTP client used for external embedding requests.</param>
        /// <param name="logger">The logger instance used for diagnostics.</param>
        /// <exception cref="ArgumentNullException">
        /// Thrown when <paramref name="http"/> or <paramref name="logger"/> is null.
        /// </exception>
        public VectorMemoryProvider(HttpClient http, ILogger<VectorMemoryProvider> logger)
        {
            _http = http ?? throw new ArgumentNullException(nameof(http));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Generates an embedding vector for the given text input.
        /// </summary>
        /// <param name="text">The text to embed.</param>
        /// <param name="ct">A cancellation token for the asynchronous operation.</param>
        /// <returns>An <see cref="Embedding"/> representing the generated vector.</returns>
        public async Task<Embedding> EmbedAsync(string text, CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Input text cannot be null or empty.", nameof(text));

            _logger.LogDebug("[Vector] Requesting embedding for input of length {Length}", text.Length);

            var payload = new { input = text, model = "text-embedding-3-large" };

            var response = await _http.PostAsJsonAsync("/embeddings", payload, ct)
                                      .ConfigureAwait(false);

            response.EnsureSuccessStatusCode();

            var vector = await response.Content
                .ReadFromJsonAsync<float[]>(cancellationToken: ct)
                .ConfigureAwait(false) ?? Array.Empty<float>();

            var embedding = new Embedding
            {
                SourceText = text,
                Vector = vector,
                Model = payload.model,
                CreatedAt = DateTimeOffset.UtcNow
            };

            _logger.LogInformation("[Vector] Generated {Dim}-dimensional embedding.", embedding.Vector.ToArray().Length);
            return embedding;
        }

        /// <summary>
        /// Stores a memory record in the in-memory vector store.
        /// </summary>
        /// <param name="record">The record to store.</param>
        /// <param name="ct">A cancellation token for the asynchronous operation.</param>
        public Task StoreAsync(MemoryRecord record, CancellationToken ct)
        {
            if (record == null)
                throw new ArgumentNullException(nameof(record));

            _logger.LogDebug("[Vector] Storing record '{Id}'", record.Id);
            _store[record.Id] = record;
            return Task.CompletedTask;
        }

        /// <summary>
        /// Performs a cosine similarity search across stored embeddings.
        /// </summary>
        /// <param name="query">The text query to embed and search for.</param>
        /// <param name="limit">The maximum number of results to return.</param>
        /// <param name="ct">A cancellation token for the asynchronous operation.</param>
        /// <returns>A collection of matching <see cref="MemoryRecord"/>s sorted by similarity.</returns>
        public async Task<IEnumerable<MemoryRecord>> SearchAsync(string query, int limit, CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(query))
                return Enumerable.Empty<MemoryRecord>();

            var queryEmbedding = await EmbedAsync(query, ct).ConfigureAwait(false);

            var results = _store.Values
                .Select(r =>
                {
                    var a = r.Embedding ?? Array.Empty<float>();
                    var b = queryEmbedding.Vector ?? Array.Empty<float>();
                    r.Similarity = CosineSimilarity(a.ToArray(), b.ToArray());
                    return r;
                })
                .OrderByDescending(r => r.Similarity)
                .Take(limit)
                .ToList();

            _logger.LogInformation("[Vector] Search completed — {Count} matches for '{Query}'", results.Count, query);
            return results;
        }

        /// <summary>
        /// Deletes a vector from the in-memory store.
        /// </summary>
        /// <param name="id">The ID of the embedding to delete.</param>
        /// <param name="ct">A cancellation token for the asynchronous operation.</param>
        public Task DeleteAsync(string id, CancellationToken ct)
        {
            _logger.LogDebug("[Vector] Deleting embedding with id {Id}", id);
            _store.Remove(id);
            return Task.CompletedTask;
        }

        /// <summary>
        /// Computes cosine similarity between two embedding vectors.
        /// </summary>
        /// <param name="a">The first vector.</param>
        /// <param name="b">The second vector.</param>
        /// <returns>A similarity score between 0 and 1.</returns>
        private static float CosineSimilarity(float[] a, float[] b)
        {
            if (a == null || b == null || a.Length == 0 || b.Length == 0 || a.Length != b.Length)
                return 0f;

            float dot = 0, magA = 0, magB = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                magA += a[i] * a[i];
                magB += b[i] * b[i];
            }

            return dot / ((float)Math.Sqrt(magA) * (float)Math.Sqrt(magB) + 1e-9f);
        }
    }
}
