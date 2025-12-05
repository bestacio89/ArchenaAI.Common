using Franz.Common.Mediator.Pipelines.Core;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Pipelines
{
    /// <summary>
    /// Defines a Franz-compatible executor that orchestrates semantic kernel pipeline operations.
    /// </summary>
    public interface IPipelineExecutor : IPipeline<Func<Task>, Task>
    {
        /// <summary>
        /// Executes a semantic operation that returns a result within the kernel pipeline.
        /// </summary>
        /// <typeparam name="T">The result type returned by the semantic operation.</typeparam>
        /// <param name="executeAsync">A delegate representing the semantic operation to execute.</param>
        /// <param name="cancellationToken">A token used to cancel the operation.</param>
        /// <returns>
        /// A task representing the asynchronous operation, containing the result of the semantic execution.
        /// </returns>
        Task<T> ExecuteAsync<T>(Func<Task<T>> executeAsync, CancellationToken cancellationToken = default);
    }
}
