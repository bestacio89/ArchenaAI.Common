using ArchenaAI.Common.Semantic.Kernel.Memory.Models;
using ArchenaAI.Common.Semantic.Kernel.Memory.Contracts;
using Microsoft.Extensions.Logging;

namespace ArchenaAI.Common.Semantic.Kernel.Memory
{
    /// <summary>
    /// Manages both vector-based (semantic) and symbolic (structured) memory layers.
    /// Combines embedding generation, symbolic storage, and hybrid search operations.
    /// </summary>
    public sealed class MemoryManager : IMemoryManager
    {
        private readonly IVectorMemoryProvider _vector;
        private readonly ISymbolicMemoryProvider _symbolic;
        private readonly ILogger<MemoryManager> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="MemoryManager"/> class.
        /// </summary>
        /// <param name="vector">The vector memory provider responsible for embedding and semantic similarity.</param>
        /// <param name="symbolic">The symbolic memory provider that handles structured and metadata-based storage.</param>
        /// <param name="logger">Logger instance used for diagnostics and tracing.</param>
        /// <exception cref="ArgumentNullException">
        /// Thrown when any of the provided dependencies are <see langword="null"/>.
        /// </exception>
        public MemoryManager(
            IVectorMemoryProvider vector,
            ISymbolicMemoryProvider symbolic,
            ILogger<MemoryManager> logger)
        {
            _vector = vector ?? throw new ArgumentNullException(nameof(vector));
            _symbolic = symbolic ?? throw new ArgumentNullException(nameof(symbolic));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Stores a new memory entry by embedding its text and saving it in both layers.
        /// </summary>
        /// <param name="id">The unique identifier of the record to store.</param>
        /// <param name="text">The raw text content to embed and persist.</param>
        /// <param name="ct">Cancellation token for the asynchronous operation.</param>
        public async Task StoreAsync(string id, string text, CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Memory text cannot be null or empty.", nameof(text));

            _logger.LogDebug("[Memory] Embedding record '{Id}'...", id);

            // Create semantic vector
            var embedding = await _vector.EmbedAsync(text, ct).ConfigureAwait(false);
            embedding.SourceText = text;

            // Create unified memory record
            var record = new MemoryRecord
            {
                Id = id,
                Content = text,
                Embedding = embedding.Vector.ToArray()
            }
            .AddMetadata("model", embedding.Model)
            .AddMetadata("createdAt", embedding.CreatedAt);


            await _symbolic.StoreAsync(record, ct).ConfigureAwait(false);

            _logger.LogInformation("[Memory] Stored record '{Id}' ({Dim}D embedding).", id, embedding.Vector.ToArray().Length);
        }

        /// <summary>
        /// Performs a semantic search combining vector similarity with symbolic filters.
        /// </summary>
        /// <param name="query">The text query or semantic prompt to search for.</param>
        /// <param name="limit">The maximum number of results to return.</param>
        /// <param name="ct">Cancellation token for the asynchronous operation.</param>
        /// <returns>A collection of memory records ordered by semantic similarity.</returns>
        public async Task<IEnumerable<MemoryRecord>> SearchAsync(string query, int limit, CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(query))
                return Enumerable.Empty<MemoryRecord>();

            _logger.LogDebug("[Memory] Searching '{Query}' (top {Limit})...", query, limit);

            var embedding = await _vector.EmbedAsync(query, ct).ConfigureAwait(false);
            var results = await _symbolic.SearchAsync(embedding.Vector.ToArray(), limit, ct).ConfigureAwait(false);

            _logger.LogInformation("[Memory] Retrieved {Count} records for query '{Query}'.", results.Count(), query);

            return results.OrderByDescending(r => r.Similarity ?? 0);
        }

        /// <summary>
        /// Removes a record from memory (both vector and symbolic layers).
        /// </summary>
        /// <param name="id">The unique identifier of the record to delete.</param>
        /// <param name="ct">Cancellation token for the asynchronous operation.</param>
        public async Task DeleteAsync(string id, CancellationToken ct)
        {
            _logger.LogDebug("[Memory] Deleting record '{Id}'...", id);
            await _symbolic.DeleteAsync(id, ct).ConfigureAwait(false);
            await _vector.DeleteAsync(id, ct).ConfigureAwait(false);
        }
    }
}
