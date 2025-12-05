using System;
using System.Threading;
using System.Threading.Tasks;
using ArchenaAI.Common.Semantic.Kernel.Pipelines;
using ArchenaAI.Common.Semantic.Kernel.Planning;
using Microsoft.Extensions.Logging;

namespace ArchenaAI.Common.Semantic.Kernel.Actions
{
    /// <summary>
    /// Kernel pipeline behavior that post-processes LLM outputs
    /// and executes structured action calls via <see cref="LLMActionPlanner"/>.
    ///
    /// This is the "Franz-style" enforcement layer: the LLM only
    /// proposes actions, the pipeline decides if and how they run.
    /// </summary>
    public sealed class LLMActionPipelineBehavior : IKernelPipelineBehavior
    {
        private readonly LLMActionPlanner _planner;
        private readonly ILogger<LLMActionPipelineBehavior> _logger;

        public LLMActionPipelineBehavior(
            LLMActionPlanner planner,
            ILogger<LLMActionPipelineBehavior> logger)
        {
            _planner = planner ?? throw new ArgumentNullException(nameof(planner));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Non-returning operations – we don't touch them for now.
        /// </summary>
        public Task HandleAsync(Func<Task> execute, CancellationToken cancellationToken = default)
        {
            if (execute is null)
                throw new ArgumentNullException(nameof(execute));

            return execute();
        }

        /// <summary>
        /// Intercepts operations that return a value. If the result is a string,
        /// we give <see cref="LLMActionPlanner"/> a chance to extract and execute
        /// a tool call and return the tool result instead.
        /// </summary>
        public async Task<T> HandleAsync<T>(
            Func<Task<T>> execute,
            CancellationToken cancellationToken = default)
        {
            if (execute is null)
                throw new ArgumentNullException(nameof(execute));

            // Execute downstream (LLM call, skill, etc.)
            var result = await execute().ConfigureAwait(false);

            // We only care about string-based LLM outputs at this layer.
            if (result is not string raw)
                return result;

            try
            {
                // For non-agent flows we don't have a real correlation id yet.
                // We still generate one so everything stays traceable.
                var correlationId = Guid.NewGuid().ToString("N");

                _logger.LogDebug(
                    "[LLMActionPipeline] Inspecting LLM output for action calls (corr: {CorrelationId})",
                    correlationId);

                var processed = await _planner
                    .ProcessAsync(raw, correlationId, cancellationToken)
                    .ConfigureAwait(false);

                // If nothing was changed, keep original result.
                if (ReferenceEquals(processed, raw))
                    return result;

                _logger.LogInformation(
                    "[LLMActionPipeline] Action successfully executed (corr: {CorrelationId})",
                    correlationId);

                // If the caller expects a string, return the processed tool result.
                if (typeof(T) == typeof(string))
                    return (T)(object)processed;

                // For now we only support string pipelines; anything else falls back.
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex,
                    "[LLMActionPipeline] Failed while processing action call. Falling back to raw LLM output.");

                return result;
            }
        }
    }
}
