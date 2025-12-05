using ArchenaAI.Common.Semantic.Kernel.Memory.Models;

namespace ArchenaAI.Common.Semantic.Kernel.Memory.Contracts
{
    /// <summary>
    /// Defines a unified contract for managing hybrid memory operations,
    /// combining symbolic (structured) and vector (semantic) storage layers.
    /// </summary>
    public interface IMemoryManager
    {
        /// <summary>
        /// Stores a new memory entry by generating its embedding
        /// and persisting it across all configured memory layers.
        /// </summary>
        /// <param name="id">The unique identifier to assign to the memory record.</param>
        /// <param name="text">The text content to embed and store.</param>
        /// <param name="ct">A token to observe while waiting for the asynchronous operation to complete.</param>
        /// <returns>
        /// A task representing the asynchronous storage operation.
        /// </returns>
        /// <exception cref="ArgumentException">
        /// Thrown when <paramref name="text"/> is null, empty, or consists only of whitespace.
        /// </exception>
        Task StoreAsync(string id, string text, CancellationToken ct);

        /// <summary>
        /// Performs a hybrid semantic search that combines vector similarity
        /// and symbolic metadata filtering.
        /// </summary>
        /// <param name="query">The textual query or keyword to search for.</param>
        /// <param name="limit">The maximum number of records to return.</param>
        /// <param name="ct">A token to observe while waiting for the asynchronous operation to complete.</param>
        /// <returns>
        /// A collection of <see cref="MemoryRecord"/> instances ranked by semantic relevance.
        /// </returns>
        /// <exception cref="ArgumentException">
        /// Thrown when <paramref name="query"/> is null, empty, or consists only of whitespace.
        /// </exception>
        Task<IEnumerable<MemoryRecord>> SearchAsync(string query, int limit, CancellationToken ct);

        /// <summary>
        /// Deletes a memory record from both vector and symbolic memory layers.
        /// </summary>
        /// <param name="id">The unique identifier of the record to delete.</param>
        /// <param name="ct">A token to observe while waiting for the asynchronous operation to complete.</param>
        /// <returns>A task representing the asynchronous deletion operation.</returns>
        /// <exception cref="ArgumentException">
        /// Thrown when <paramref name="id"/> is null, empty, or consists only of whitespace.
        /// </exception>
        Task DeleteAsync(string id, CancellationToken ct);
    }
}
