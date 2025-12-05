using Microsoft.Extensions.Logging;
using Polly;
using Polly.Timeout;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Pipelines.Behaviours
{
    /// <summary>
    /// Provides operation-level timeout protection for semantic kernel stages.
    /// This is distinct from the global timeout used in ResilienceBehavior.
    ///
    /// TimeoutBehavior ensures that individual skills or internal LLM calls
    /// cannot block the pipeline indefinitely.
    /// </summary>
    public sealed class TimeoutBehavior : KernelPipelineBehavior
    {
        private readonly AsyncTimeoutPolicy _timeoutPolicy;

        public TimeoutBehavior(ILogger<TimeoutBehavior> logger)
            : base(logger)
        {
            if (logger == null)
                throw new ArgumentNullException(nameof(logger));

            _timeoutPolicy = Policy.TimeoutAsync(TimeSpan.FromSeconds(4));
        }

        public override async Task HandleAsync(Func<Task> execute, CancellationToken cancellationToken = default)
        {
            try
            {
                await _timeoutPolicy.ExecuteAsync(
                    ct => base.HandleAsync(execute, ct),
                    cancellationToken).ConfigureAwait(false);
            }
            catch (TimeoutRejectedException)
            {
                Logger.LogError("[TimeoutBehavior] Operation timed out.");
                throw;
            }
        }

        public override async Task<T> HandleAsync<T>(Func<Task<T>> execute, CancellationToken cancellationToken = default)
        {
            try
            {
                return await _timeoutPolicy.ExecuteAsync(
                    ct => base.HandleAsync(execute, ct),
                    cancellationToken).ConfigureAwait(false);
            }
            catch (TimeoutRejectedException)
            {
                Logger.LogError("[TimeoutBehavior] Operation timed out.");
                throw;
            }
        }
    }
}
