using ArchenaAI.Common.Semantic.Kernel.Memory.Contracts;
using ArchenaAI.Common.Semantic.Kernel.Memory.Models;
using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;

namespace ArchenaAI.Common.Semantic.Kernel.Memory
{
    /// <summary>
    /// Default in-memory symbolic memory provider.
    /// Stores structured knowledge (facts, metadata, or contextual records)
    /// that complement the vector memory.
    /// </summary>
    public sealed class SymbolicMemoryProvider : ISymbolicMemoryProvider
    {
        private readonly ConcurrentDictionary<string, MemoryRecord> _memory;
        private readonly ILogger<SymbolicMemoryProvider> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="SymbolicMemoryProvider"/> class.
        /// </summary>
        /// <param name="logger">Logger instance for diagnostic output.</param>
        public SymbolicMemoryProvider(ILogger<SymbolicMemoryProvider> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _memory = new ConcurrentDictionary<string, MemoryRecord>();
        }

        /// <summary>
        /// Stores or updates a symbolic memory record.
        /// </summary>
        public Task StoreAsync(MemoryRecord record, CancellationToken ct)
        {
            ArgumentNullException.ThrowIfNull(record);

            _memory[record.Id] = record;

            _logger.LogDebug("[Symbolic] Stored record '{Id}' (metadata: {Keys})",
                record.Id, string.Join(",", record.Metadata.Keys));

            return Task.CompletedTask;
        }

        /// <summary>
        /// Retrieves a memory record by its key.
        /// </summary>
        public Task<MemoryRecord?> GetByKeyAsync(string key, CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(key))
                throw new ArgumentException("Key cannot be null or empty.", nameof(key));

            _memory.TryGetValue(key, out var record);

            _logger.LogDebug(record != null
                ? "[Symbolic] Retrieved record '{Key}'"
                : "[Symbolic] No record found for key '{Key}'", key);

            return Task.FromResult(record);
        }

        /// <summary>
        /// Searches stored memory records using metadata or symbolic keyword matching.
        /// </summary>
        /// <param name="query">A text query or keyword to search for in metadata.</param>
        /// <param name="limit">Maximum number of results to return.</param>
        /// <param name="ct">Cancellation token for async operations.</param>
        public Task<IEnumerable<MemoryRecord>> SearchAsync(string query, int limit, CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(query))
                return Task.FromResult<IEnumerable<MemoryRecord>>(Array.Empty<MemoryRecord>());

            var results = _memory.Values
                .Where(r => r.Metadata.Values
                    .Any(v => v?.ToString()?.Contains(query, StringComparison.OrdinalIgnoreCase) ?? false))
                .Take(limit)
                .ToList();

            _logger.LogInformation("[Symbolic] Metadata search found {Count} matches for '{Query}'.", results.Count, query);
            return Task.FromResult<IEnumerable<MemoryRecord>>(results);
        }

        /// <summary>
        /// Searches stored memory records by vector similarity (symbolic embedding matching).
        /// </summary>
        /// <param name="vector">Input vector for similarity comparison.</param>
        /// <param name="limit">Maximum number of results to return.</param>
        /// <param name="ct">Cancellation token for async operations.</param>
        public Task<IEnumerable<MemoryRecord>> SearchAsync(float[] vector, int limit, CancellationToken ct)
        {
            if (vector is null || vector.Length == 0 || _memory.IsEmpty)
                return Task.FromResult<IEnumerable<MemoryRecord>>(Array.Empty<MemoryRecord>());

            var results = _memory.Values
                .Select(record =>
                {
                    record.Similarity = ComputeSimilarity(record.Embedding?.ToArray(), vector);
                    return record;
                })
                .OrderByDescending(r => r.Similarity)
                .Take(limit)
                .ToList();

            _logger.LogInformation("[Symbolic] Vector similarity search returned {Count} results.", results.Count);
            return Task.FromResult<IEnumerable<MemoryRecord>>(results);
        }

        /// <summary>
        /// Removes a record by ID.
        /// </summary>
        public Task DeleteAsync(string id, CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(id))
                throw new ArgumentException("ID cannot be null or empty.", nameof(id));

            _memory.TryRemove(id, out _);
            _logger.LogDebug("[Symbolic] Deleted record '{Id}'", id);

            return Task.CompletedTask;
        }

        /// <summary>
        /// Computes cosine similarity between two vectors.
        /// </summary>
        private static double ComputeSimilarity(float[]? v1, float[]? v2)
        {
            if (v1 is null || v2 is null || v1.Length != v2.Length)
                return 0.0;

            double dot = 0, normA = 0, normB = 0;
            for (int i = 0; i < v1.Length; i++)
            {
                dot += v1[i] * v2[i];
                normA += v1[i] * v1[i];
                normB += v2[i] * v2[i];
            }

            return normA == 0 || normB == 0 ? 0.0 : dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        }
    }
}
