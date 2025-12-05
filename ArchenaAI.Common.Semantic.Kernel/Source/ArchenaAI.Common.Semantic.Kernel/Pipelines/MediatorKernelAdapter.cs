using Franz.Common.Mediator.Pipelines.Core;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Pipelines
{
    /// <summary>
    /// Bridge that connects Franz Mediator pipelines with ArchenaAI Semantic Kernel behaviors.
    /// </summary>
    /// <typeparam name="TRequest">The mediator request type.</typeparam>
    /// <typeparam name="TResponse">The mediator response type.</typeparam>
    public sealed class MediatorKernelAdapter<TRequest, TResponse> : IPipeline<TRequest, TResponse>
    {
        private readonly IEnumerable<IKernelPipelineBehavior> _kernelBehaviors;

        /// <summary>
        /// Initializes a new instance of the <see cref="MediatorKernelAdapter{TRequest, TResponse}"/> class.
        /// </summary>
        /// <param name="kernelBehaviors">A collection of semantic kernel behaviors to execute.</param>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="kernelBehaviors"/> is null.</exception>
        public MediatorKernelAdapter(IEnumerable<IKernelPipelineBehavior> kernelBehaviors)
        {
            _kernelBehaviors = kernelBehaviors ?? throw new ArgumentNullException(nameof(kernelBehaviors));
        }

        /// <summary>
        /// Executes a Mediator request through the Semantic Kernel pipeline chain.
        /// </summary>
        /// <param name="request">The mediator request to process.</param>
        /// <param name="next">A delegate that invokes the next mediator handler in the chain.</param>
        /// <param name="cancellationToken">A cancellation token to cancel the operation.</param>
        /// <returns>A task representing the asynchronous operation, containing the mediator response.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="next"/> is null.</exception>
        public async Task<TResponse> Handle(
            TRequest request,
            Func<Task<TResponse>> next,
            CancellationToken cancellationToken = default)
        {
            if (next is null)
                throw new ArgumentNullException(nameof(next));

            Task<TResponse> InvokeNext() => next();

            // Execute each semantic kernel behavior sequentially
            foreach (var behavior in _kernelBehaviors)
            {
                await behavior
                    .HandleAsync(() => InvokeNext(), cancellationToken)
                    .ConfigureAwait(false);
            }

            // Execute the actual mediator handler
            return await InvokeNext().ConfigureAwait(false);
        }
    }
}
