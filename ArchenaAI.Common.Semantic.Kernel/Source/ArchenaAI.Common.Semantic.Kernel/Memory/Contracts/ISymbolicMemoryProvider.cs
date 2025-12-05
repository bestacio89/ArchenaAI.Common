using ArchenaAI.Common.Semantic.Kernel.Memory.Models;

namespace ArchenaAI.Common.Semantic.Kernel.Memory.Contracts
{
    /// <summary>
    /// Defines operations for managing symbolic (structured) memory,
    /// typically used for persistent factual or contextual information.
    /// </summary>
    public interface ISymbolicMemoryProvider : IMemoryProvider
    {
        /// <summary>
        /// Stores or updates a memory record in the symbolic store.
        /// </summary>
        new Task StoreAsync(MemoryRecord record, CancellationToken ct);

        /// <summary>
        /// Retrieves a memory record by its unique key.
        /// </summary>
        /// <param name="key">The unique identifier of the record to retrieve.</param>
        /// <param name="ct">A cancellation token to observe while waiting for the task to complete.</param>
        /// <returns>
        /// The <see cref="MemoryRecord"/> matching the given key, or <see langword="null"/> if not found.
        /// </returns>
        Task<MemoryRecord?> GetByKeyAsync(string key, CancellationToken ct);

        /// <summary>
        /// Performs a semantic search using a query vector,
        /// optionally limited to the top N results.
        /// </summary>
        /// <param name="vector">The embedding vector used for symbolic similarity search.</param>
        /// <param name="limit">The maximum number of records to return.</param>
        /// <param name="ct">A cancellation token to observe while waiting for the task to complete.</param>
        /// <returns>A collection of <see cref="MemoryRecord"/> instances ordered by similarity.</returns>
        Task<IEnumerable<MemoryRecord>> SearchAsync(float[] vector, int limit, CancellationToken ct);

        /// <summary>
        /// Deletes a memory record from the symbolic layer.
        /// </summary>
        /// <param name="id">The identifier of the record to delete.</param>
        /// <param name="ct">A cancellation token to observe while waiting for the task to complete.</param>
        Task DeleteAsync(string id, CancellationToken ct);
    }
}
