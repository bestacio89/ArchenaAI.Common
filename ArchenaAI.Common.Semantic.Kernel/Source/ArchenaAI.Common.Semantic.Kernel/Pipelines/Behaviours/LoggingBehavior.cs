using Microsoft.Extensions.Logging;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Pipelines.Behaviours
{
    /// <summary>
    /// Logs execution details of semantic kernel pipeline stages.
    /// This is the semantic counterpart of Franz.Common.Mediator.LoggingPipelineBehavior.
    /// </summary>
    public sealed class LoggingBehavior : KernelPipelineBehavior
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LoggingBehavior"/> class.
        /// </summary>
        /// <param name="logger">The logger instance used for diagnostic messages.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="logger"/> is null.</exception>
        public LoggingBehavior(ILogger<LoggingBehavior> logger)
            : base(logger)
        {
        }

        /// <summary>
        /// Called before executing the next pipeline stage.
        /// Logs the start of a semantic reasoning phase.
        /// </summary>
        /// <param name="cancellationToken">A token used to cancel the operation.</param>
        protected override async Task OnBeforeAsync(CancellationToken cancellationToken)
        {
            Logger.LogInformation("[SemanticKernel] Starting reasoning stage at {Timestamp}", DateTimeOffset.UtcNow);
            await Task.CompletedTask.ConfigureAwait(false);
        }

        /// <summary>
        /// Called after executing the next pipeline stage.
        /// Logs the completion of a semantic reasoning phase.
        /// </summary>
        /// <param name="cancellationToken">A token used to cancel the operation.</param>
        protected override async Task OnAfterAsync(CancellationToken cancellationToken)
        {
            Logger.LogInformation("[SemanticKernel] Finished reasoning stage at {Timestamp}", DateTimeOffset.UtcNow);
            await Task.CompletedTask.ConfigureAwait(false);
        }
    }
}
