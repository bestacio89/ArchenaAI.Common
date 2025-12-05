using ArchenaAI.Common.Semantic.Kernel.Memory.Models;

namespace ArchenaAI.Common.Semantic.Kernel.Memory.Contracts
{
    /// <summary>
    /// Defines the base contract for interacting with memory storage layers.
    /// Provides common operations for storing and retrieving semantic or symbolic records.
    /// </summary>
    public interface IMemoryProvider
    {
        /// <summary>
        /// Stores or updates a memory record in the provider.
        /// Implementations may persist the record in symbolic, vector, or hybrid memory layers.
        /// </summary>
        /// <param name="record">The <see cref="MemoryRecord"/> to store or update.</param>
        /// <param name="ct">A token to observe while waiting for the asynchronous operation to complete.</param>
        /// <returns>
        /// A task representing the asynchronous store operation.
        /// </returns>
        /// <exception cref="ArgumentNullException">
        /// Thrown when <paramref name="record"/> is <see langword="null"/>.
        /// </exception>
        Task StoreAsync(MemoryRecord record, CancellationToken ct);

        /// <summary>
        /// Performs a text-based or metadata-based search within the memory provider.
        /// </summary>
        /// <param name="query">The textual query or keyword to match against memory records.</param>
        /// <param name="limit">The maximum number of records to return.</param>
        /// <param name="ct">A token to observe while waiting for the asynchronous operation to complete.</param>
        /// <returns>
        /// A collection of <see cref="MemoryRecord"/> instances matching the query,
        /// typically ordered by relevance or similarity.
        /// </returns>
        /// <exception cref="ArgumentException">
        /// Thrown when <paramref name="query"/> is null, empty, or whitespace.
        /// </exception>
        Task<IEnumerable<MemoryRecord>> SearchAsync(string query, int limit, CancellationToken ct);
    }
}
