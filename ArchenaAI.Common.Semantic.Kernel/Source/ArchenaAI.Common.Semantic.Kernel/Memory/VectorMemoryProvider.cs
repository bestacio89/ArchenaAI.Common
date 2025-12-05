using ArchenaAI.Common.Semantic.Kernel.Memory.Contracts;
using ArchenaAI.Common.Semantic.Kernel.Memory.Models;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Memory
{
    public sealed class VectorMemoryProvider : IVectorMemoryProvider
    {
        private const string DefaultModel = "text-embedding-3-large";

        private readonly HttpClient _http;
        private readonly ILogger<VectorMemoryProvider> _logger;
        private readonly string _modelName;

        private readonly ConcurrentDictionary<string, MemoryRecord> _store = new();

        public VectorMemoryProvider(
            HttpClient http,
            ILogger<VectorMemoryProvider> logger,
            IConfiguration configuration)
        {
            _http = http ?? throw new ArgumentNullException(nameof(http));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));

            _modelName = configuration["Kernel:Memory:EmbeddingModel"];
            if (string.IsNullOrWhiteSpace(_modelName))
                _modelName = DefaultModel;

            _logger.LogInformation("[Vector] Using embedding model '{Model}'", _modelName);
        }

        public async Task<Embedding> EmbedAsync(string text, CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Input text cannot be null or empty.", nameof(text));

            _logger.LogDebug("[Vector] Requesting embedding for input length {Length}", text.Length);

            var payload = new { input = text, model = _modelName };

            using var response = await _http.PostAsJsonAsync("/embeddings", payload, ct)
                                            .ConfigureAwait(false);

            response.EnsureSuccessStatusCode();

            var vector = await response.Content
                .ReadFromJsonAsync<float[]>(cancellationToken: ct)
                .ConfigureAwait(false) ?? Array.Empty<float>();

            var embedding = new Embedding
            {
                SourceText = text,
                Vector = vector,                 // <-- IReadOnlyList<float>
                Model = payload.model,
                CreatedAt = DateTimeOffset.UtcNow
            };

            _logger.LogInformation("[Vector] Generated {Dim}-D embedding.", embedding.Vector.Count);
            return embedding;
        }

        public Task StoreAsync(MemoryRecord record, CancellationToken ct)
        {
            if (record is null)
                throw new ArgumentNullException(nameof(record));

            _logger.LogDebug("[Vector] Storing record '{Id}'", record.Id);
            _store[record.Id] = record;
            return Task.CompletedTask;
        }

        public async Task<IEnumerable<MemoryRecord>> SearchAsync(string query, int limit, CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(query))
                return Enumerable.Empty<MemoryRecord>();

            if (_store.IsEmpty)
                return Enumerable.Empty<MemoryRecord>();

            var queryEmbedding = await EmbedAsync(query, ct).ConfigureAwait(false);

            var queryVec = queryEmbedding.Vector?.ToArray() ?? Array.Empty<float>();

            var results = _store.Values
                .Select(r =>
                {
                    var a = r.Embedding?.ToArray() ?? Array.Empty<float>();
                    var similarity = CosineSimilarity(a, queryVec);
                    r.Similarity = similarity;
                    return r;
                })
                .OrderByDescending(r => r.Similarity ?? 0f)
                .Take(limit)
                .ToList();

            return results;
        }

        public Task DeleteAsync(string id, CancellationToken ct)
        {
            _store.TryRemove(id, out _);
            return Task.CompletedTask;
        }

        private static float CosineSimilarity(float[] a, float[] b)
        {
            if (a == null || b == null || a.Length == 0 || b.Length == 0 || a.Length != b.Length)
                return 0f;

            // SIMD path
            if (Vector.IsHardwareAccelerated && a.Length >= Vector<float>.Count)
            {
                int i = 0;
                var dotVec = Vector<float>.Zero;
                var magAVec = Vector<float>.Zero;
                var magBVec = Vector<float>.Zero;

                int simdLen = a.Length - (a.Length % Vector<float>.Count);

                for (; i < simdLen; i += Vector<float>.Count)
                {
                    var va = new Vector<float>(a, i);
                    var vb = new Vector<float>(b, i);

                    dotVec += va * vb;
                    magAVec += va * va;
                    magBVec += vb * vb;
                }

                float dot = 0, magA = 0, magB = 0;

                for (int j = 0; j < Vector<float>.Count; j++)
                {
                    dot += dotVec[j];
                    magA += magAVec[j];
                    magB += magBVec[j];
                }

                for (; i < a.Length; i++)
                {
                    dot += a[i] * b[i];
                    magA += a[i] * a[i];
                    magB += b[i] * b[i];
                }

                return dot / ((float)Math.Sqrt(magA) * (float)Math.Sqrt(magB) + 1e-9f);
            }

            // Scalar fallback
            float dotScalar = 0, magAScalar = 0, magBScalar = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dotScalar += a[i] * b[i];
                magAScalar += a[i] * a[i];
                magBScalar += b[i] * b[i];
            }

            return dotScalar / ((float)Math.Sqrt(magAScalar) * (float)Math.Sqrt(magBScalar) + 1e-9f);
        }
    }
}
