using ArchenaAI.Common.Semantic.Kernel.Memory.Contracts;
using Microsoft.Extensions.Logging;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Pipelines
{
    /// <summary>
    /// Injects vector memory or embeddings into the Semantic Kernel context.
    /// </summary>
    public sealed class VectorMemoryKernelBehavior : KernelPipelineBehavior
    {
        private readonly IVectorMemoryProvider _vectorMemory;

        /// <summary>
        /// Initializes a new instance of the <see cref="VectorMemoryKernelBehavior"/> class.
        /// </summary>
        /// <param name="vectorMemory">The vector memory provider used to generate and retrieve embeddings.</param>
        /// <param name="logger">The logger instance used for diagnostics.</param>
        /// <exception cref="ArgumentNullException">
        /// Thrown when <paramref name="vectorMemory"/> or <paramref name="logger"/> is null.
        /// </exception>
        public VectorMemoryKernelBehavior(
            IVectorMemoryProvider vectorMemory,
            ILogger<VectorMemoryKernelBehavior> logger)
            : base(logger)
        {
            _vectorMemory = vectorMemory ?? throw new ArgumentNullException(nameof(vectorMemory));
        }

        /// <summary>
        /// Called before executing the next pipeline stage.
        /// Preloads or injects vector-based context into the reasoning process.
        /// </summary>
        /// <param name="cancellationToken">A token used to cancel the operation.</param>
        protected override async Task OnBeforeAsync(CancellationToken cancellationToken)
        {
            Logger.LogDebug("[KernelPipeline] Injecting vector memory context...");
            // Future: preload embeddings or cache semantic context before reasoning.
            await Task.CompletedTask.ConfigureAwait(false);
        }

        /// <summary>
        /// Called after executing the next pipeline stage.
        /// Allows post-processing or storing of vector-based results if necessary.
        /// </summary>
        /// <param name="cancellationToken">A token used to cancel the operation.</param>
        protected override async Task OnAfterAsync(CancellationToken cancellationToken)
        {
            Logger.LogDebug("[KernelPipeline] Finalizing vector memory context...");
            await Task.CompletedTask.ConfigureAwait(false);
        }
    }
}
