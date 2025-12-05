using Microsoft.Extensions.Logging;
using Polly;
using Polly.Retry;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Pipelines.Behaviours
{
    /// <summary>
    /// Provides fine-grained retry logic around semantic kernel operations.
    /// Unlike ResilienceBehavior (which handles global pipeline resilience),
    /// this behavior focuses on retrying *skill-level* transient failures,
    /// such as malformed JSON responses, intermittent API issues, or temporary
    /// dependency failures.
    /// </summary>
    public sealed class RetryBehavior : KernelPipelineBehavior
    {
        private readonly AsyncRetryPolicy _retryPolicy;

        public RetryBehavior(ILogger<RetryBehavior> logger)
            : base(logger)
        {
            if (logger == null)
                throw new ArgumentNullException(nameof(logger));

            _retryPolicy = Policy
                .Handle<Exception>()
                .WaitAndRetryAsync(
                    retryCount: 2,
                    sleepDurationProvider: attempt => TimeSpan.FromMilliseconds(150 * attempt),
                    onRetry: (ex, delay, attempt, _) =>
                    {
                        Logger.LogWarning(
                            "[RetryBehavior] Retry #{Attempt} after {Delay}ms due to: {Error}",
                            attempt,
                            delay.TotalMilliseconds,
                            ex.Message);
                    });
        }

        public override async Task HandleAsync(Func<Task> execute, CancellationToken cancellationToken = default)
        {
            await _retryPolicy.ExecuteAsync(() => base.HandleAsync(execute, cancellationToken))
                              .ConfigureAwait(false);
        }

        public override async Task<T> HandleAsync<T>(Func<Task<T>> execute, CancellationToken cancellationToken = default)
        {
            return await _retryPolicy.ExecuteAsync(() => base.HandleAsync(execute, cancellationToken))
                                     .ConfigureAwait(false);
        }
    }
}
