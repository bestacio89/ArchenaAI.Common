using Franz.Common.Mediator.Pipelines.Core;
using Microsoft.Extensions.Logging;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Pipelines
{
    /// <summary>
    /// Base class for all Semantic Kernel pipeline behaviors.
    /// Provides before/after hooks and integrates with Franz's pipeline infrastructure.
    /// </summary>
    public abstract class KernelPipelineBehavior : IKernelPipelineBehavior
    {
        /// <summary>
        /// Gets the logger used for diagnostic and trace events.
        /// </summary>
        protected ILogger Logger { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="KernelPipelineBehavior"/> class.
        /// </summary>
        /// <param name="logger">The logger instance used by the behavior.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="logger"/> is null.</exception>
        protected KernelPipelineBehavior(ILogger logger)
        {
            Logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Executes the pipeline handler according to the Franz mediator interface.
        /// </summary>
        /// <param name="request">The request delegate to execute.</param>
        /// <param name="next">The next step delegate in the pipeline chain.</param>
        /// <param name="cancellationToken">A cancellation token for the async operation.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="request"/> or <paramref name="next"/> is null.</exception>
        public async Task Handle(Func<Task> request, Func<Func<Task>, Task> next, CancellationToken cancellationToken = default)
        {
            if (request is null)
                throw new ArgumentNullException(nameof(request));
            if (next is null)
                throw new ArgumentNullException(nameof(next));

            await HandleAsync(() => next(request), cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Handles a semantic operation that does not return a value.
        /// </summary>
        /// <param name="execute">The delegate representing the next operation to execute.</param>
        /// <param name="cancellationToken">A cancellation token for the async operation.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="execute"/> is null.</exception>
        public virtual async Task HandleAsync(Func<Task> execute, CancellationToken cancellationToken = default)
        {
            if (execute is null)
                throw new ArgumentNullException(nameof(execute));

            await OnBeforeAsync(cancellationToken).ConfigureAwait(false);
            await execute().ConfigureAwait(false);
            await OnAfterAsync(cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Handles a semantic operation that returns a result.
        /// </summary>
        /// <typeparam name="T">The type of result produced by the operation.</typeparam>
        /// <param name="execute">The delegate representing the next operation to execute.</param>
        /// <param name="cancellationToken">A cancellation token for the async operation.</param>
        /// <returns>A task representing the asynchronous operation containing the result.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="execute"/> is null.</exception>
        public virtual async Task<T> HandleAsync<T>(Func<Task<T>> execute, CancellationToken cancellationToken = default)
        {
            if (execute is null)
                throw new ArgumentNullException(nameof(execute));

            await OnBeforeAsync(cancellationToken).ConfigureAwait(false);
            var result = await execute().ConfigureAwait(false);
            await OnAfterAsync(cancellationToken).ConfigureAwait(false);
            return result;
        }

        /// <summary>
        /// Called before the next operation executes.
        /// Override this to perform setup or preconditions.
        /// </summary>
        /// <param name="cancellationToken">A cancellation token for the async operation.</param>
        /// <returns>A task representing the asynchronous pre-execution step.</returns>
        protected virtual Task OnBeforeAsync(CancellationToken cancellationToken) => Task.CompletedTask;

        /// <summary>
        /// Called after the next operation completes.
        /// Override this to perform teardown or post-processing.
        /// </summary>
        /// <param name="cancellationToken">A cancellation token for the async operation.</param>
        /// <returns>A task representing the asynchronous post-execution step.</returns>
        protected virtual Task OnAfterAsync(CancellationToken cancellationToken) => Task.CompletedTask;
    }
}
