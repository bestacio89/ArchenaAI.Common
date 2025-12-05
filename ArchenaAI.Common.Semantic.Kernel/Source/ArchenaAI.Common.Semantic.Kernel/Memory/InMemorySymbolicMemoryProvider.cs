using ArchenaAI.Common.Semantic.Kernel.Memory.Contracts;
using ArchenaAI.Common.Semantic.Kernel.Memory.Models;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Memory
{
    /// <summary>
    /// Default in-memory implementation of a symbolic memory provider.
    /// Stores structured knowledge or metadata that complements vector memory.
    /// </summary>
    public sealed class InMemorySymbolicMemoryProvider : ISymbolicMemoryProvider
    {
        private readonly List<MemoryRecord> _store = new();

        public Task StoreAsync(MemoryRecord record, CancellationToken ct)
        {
            _store.RemoveAll(r => r.Id == record.Id);
            _store.Add(record);
            return Task.CompletedTask;
        }

        /// <summary>
        /// Symbolic text-based search across content and metadata fields.
        /// </summary>
        public Task<IEnumerable<MemoryRecord>> SearchAsync(string query, int limit, CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(query))
                return Task.FromResult(Enumerable.Empty<MemoryRecord>());

            var results = _store
                .Where(r =>
                    // Search inside Content (if it's a string)
                    (!string.IsNullOrWhiteSpace(r.Content) && r.Content.Contains(query, System.StringComparison.OrdinalIgnoreCase)) ||

                    // Search inside any metadata key or value
                    (r.Metadata != null && r.Metadata.Any(kv =>
                        (kv.Key?.Contains(query, System.StringComparison.OrdinalIgnoreCase) ?? false) ||
                        (kv.Value?.ToString()?.Contains(query, System.StringComparison.OrdinalIgnoreCase) ?? false)))
                )
                .Take(limit);

            return Task.FromResult(results);
        }

        /// <summary>
        /// Vector-based search (placeholder similarity scoring).
        /// </summary>
        public Task<IEnumerable<MemoryRecord>> SearchAsync(float[] vector, int limit, CancellationToken ct)
        {
            // Placeholder: cosine similarity / nearest neighbor matching would go here
            var results = _store.Take(limit);
            return Task.FromResult(results);
        }

        /// <summary>
        /// Retrieve a specific memory record by its key (ID).
        /// </summary>
        public Task<MemoryRecord?> GetByKeyAsync(string id, CancellationToken ct)
        {
            var record = _store.FirstOrDefault(r => r.Id == id);
            return Task.FromResult(record);
        }

        public Task DeleteAsync(string id, CancellationToken ct)
        {
            _store.RemoveAll(r => r.Id == id);
            return Task.CompletedTask;
        }
    }
}
