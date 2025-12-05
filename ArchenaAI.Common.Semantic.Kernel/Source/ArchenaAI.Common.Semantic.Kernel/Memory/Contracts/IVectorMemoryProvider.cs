using ArchenaAI.Common.Semantic.Kernel.Memory.Models;

namespace ArchenaAI.Common.Semantic.Kernel.Memory.Contracts
{
    /// <summary>
    /// Defines a contract for managing vector-based (semantic) memory operations.
    /// Implementations typically interact with an external embedding model or API
    /// to transform text into numeric representations for similarity search or reasoning.
    /// </summary>
    public interface IVectorMemoryProvider : IMemoryProvider
    {
        /// <summary>
        /// Generates a vector embedding representation for the given text input.
        /// </summary>
        /// <param name="text">The text to convert into a semantic embedding.</param>
        /// <param name="ct">A token to observe while waiting for the task to complete.</param>
        /// <returns>
        /// A task that represents the asynchronous operation. 
        /// The task result contains the <see cref="Embedding"/> representing the input text.
        /// </returns>
        /// <exception cref="ArgumentException">
        /// Thrown when <paramref name="text"/> is null, empty, or whitespace.
        /// </exception>
        Task<Embedding> EmbedAsync(string text, CancellationToken ct);

        /// <summary>
        /// Deletes a stored embedding from the memory provider by its identifier.
        /// </summary>
        /// <param name="id">The unique identifier of the embedding to delete.</param>
        /// <param name="ct">A token to observe while waiting for the task to complete.</param>
        /// <returns>A task representing the asynchronous delete operation.</returns>
        /// <exception cref="ArgumentException">
        /// Thrown when <paramref name="id"/> is null, empty, or whitespace.
        /// </exception>
        Task DeleteAsync(string id, CancellationToken ct);
    }
}
