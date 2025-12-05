using System;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Pipelines
{
    /// <summary>
    /// Defines a semantic processing pipeline that operates independently
    /// of Franz's mediator pipelines. Each behavior can intercept or extend
    /// the execution flow of semantic operations.
    /// </summary>
    public interface IKernelPipelineBehavior
    {
        /// <summary>
        /// Executes a semantic operation without returning a value.
        /// </summary>
        /// <param name="execute">A delegate representing the next operation in the pipeline.</param>
        /// <param name="cancellationToken">A token used to cancel the operation.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        Task HandleAsync(Func<Task> execute, CancellationToken cancellationToken = default);

        /// <summary>
        /// Executes a semantic operation that returns a result.
        /// </summary>
        /// <typeparam name="T">The result type returned by the operation.</typeparam>
        /// <param name="execute">A delegate representing the next operation in the pipeline.</param>
        /// <param name="cancellationToken">A token used to cancel the operation.</param>
        /// <returns>A task representing the asynchronous operation, containing the result.</returns>
        Task<T> HandleAsync<T>(Func<Task<T>> execute, CancellationToken cancellationToken = default);
    }
}
