using Franz.Common.Mediator.Pipelines.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Pipelines
{
    /// <summary>
    /// Executes semantic kernel operations through the Franz-compatible pipeline.
    /// </summary>
    public sealed class PipelineExecutor : IPipelineExecutor
    {
        private readonly IKernelPipelineBehavior[] _behaviors;

        public PipelineExecutor(IEnumerable<IKernelPipelineBehavior> behaviors)
        {
            _behaviors = behaviors?.ToArray()
                ?? throw new ArgumentNullException(nameof(behaviors));
        }

        // --------------------------------------------------------------------
        // IMPLEMENTATION OF IPipeline<Func<Task>, Task>
        // --------------------------------------------------------------------
        public async Task<Task> Handle(
            Func<Task> request,
            Func<Task<Task>> next,
            CancellationToken cancellationToken = default)
        {
            if (request is null)
                throw new ArgumentNullException(nameof(request));
            if (next is null)
                throw new ArgumentNullException(nameof(next));

            // Build the SK pipeline around the request delegate
            Func<Task> pipeline = request;

            foreach (var behavior in _behaviors.Reverse())
            {
                var previous = pipeline;
                pipeline = () => behavior.HandleAsync(previous, cancellationToken);
            }

            // Execute the pipeline before the mediator next()
            await pipeline().ConfigureAwait(false);

            // Execute the mediator next delegate
            return await next().ConfigureAwait(false);
        }

        // --------------------------------------------------------------------
        // SK CUSTOM EXECUTION API
        // --------------------------------------------------------------------
        public async Task<T> ExecuteAsync<T>(
            Func<Task<T>> executeAsync,
            CancellationToken cancellationToken = default)
        {
            if (executeAsync is null)
                throw new ArgumentNullException(nameof(executeAsync));

            Func<Task> root = async () => _ = await executeAsync().ConfigureAwait(false);

            Func<Task<T>> final = executeAsync;

            // Wrap result inside a Task<Task<T>> as expected by mediator-style next()
            async Task<Task<T>> Next() => await Task.FromResult(final());

            // Run through pipeline → then return the result
            foreach (var behavior in _behaviors.Reverse())
            {
                var previous = root;
                root = () => behavior.HandleAsync(previous, cancellationToken);
            }

            await root().ConfigureAwait(false);

            return await final().ConfigureAwait(false);
        }
    }
}
