using Microsoft.Extensions.Logging;
using Polly;
using Polly.Retry;
using Polly.Timeout;
using Polly.CircuitBreaker;
using System;
using System.Diagnostics;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Pipelines.Behaviours
{
    /// <summary>
    /// Provides transient-fault handling, retry, and circuit-breaker logic
    /// for semantic kernel pipeline operations.
    /// Mirrors the resilience patterns used in Franz.Common.Mediator.
    /// </summary>
    public sealed class ResilienceBehavior : KernelPipelineBehavior
    {
        private readonly AsyncRetryPolicy _retryPolicy;
        private readonly AsyncTimeoutPolicy _timeoutPolicy;
        private readonly AsyncCircuitBreakerPolicy _breakerPolicy;
        private readonly ILogger<ResilienceBehavior> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="ResilienceBehavior"/> class.
        /// </summary>
        /// <param name="logger">The logger instance.</param>
        public ResilienceBehavior(ILogger<ResilienceBehavior> logger)
            : base(logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));

            // Retry policy with exponential backoff
            _retryPolicy = Policy
                .Handle<TimeoutException>()
                .Or<HttpRequestException>()
                .Or<TaskCanceledException>()
                .WaitAndRetryAsync(
                    retryCount: 3,
                    sleepDurationProvider: attempt => TimeSpan.FromMilliseconds(100 * Math.Pow(2, attempt)),
                    onRetry: (ex, delay, attempt, _) =>
                    {
                        _logger.LogWarning(
                            "[SemanticKernel] Retry {Attempt} after {Delay} ms due to {ErrorType}: {Message}",
                            attempt, delay.TotalMilliseconds, ex.GetType().Name, ex.Message);
                    });

            // Timeout policy
            _timeoutPolicy = Policy.TimeoutAsync(TimeSpan.FromSeconds(10));

            // Circuit-breaker policy
            _breakerPolicy = Policy
                .Handle<Exception>()
                .CircuitBreakerAsync(
                    exceptionsAllowedBeforeBreaking: 5,
                    durationOfBreak: TimeSpan.FromSeconds(30),
                    onBreak: (ex, breakDelay) =>
                        _logger.LogWarning(
                            "[SemanticKernel] Circuit open for {Delay}s: {Message}",
                            breakDelay.TotalSeconds, ex.Message),
                    onReset: () => _logger.LogInformation("[SemanticKernel] Circuit closed — normal operation resumed."));

            // Compose the policies
            _policy = Policy.WrapAsync(_breakerPolicy, _retryPolicy, _timeoutPolicy);
        }

        private readonly IAsyncPolicy _policy;

        /// <summary>
        /// Handles a non-generic semantic operation with resilience logic.
        /// </summary>
        /// <param name="next">The next delegate to execute.</param>
        /// <param name="ct">A cancellation token.</param>
        public override async Task HandleAsync(Func<Task> next, CancellationToken ct = default)
        {
            await _policy.ExecuteAsync(async () =>
            {
                await base.HandleAsync(next, ct).ConfigureAwait(false);
            }).ConfigureAwait(false);
        }

        /// <summary>
        /// Handles a generic semantic operation with resilience logic.
        /// </summary>
        /// <typeparam name="T">The result type.</typeparam>
        /// <param name="next">The next delegate to execute.</param>
        /// <param name="ct">A cancellation token.</param>
        /// <returns>The resulting value from the pipeline.</returns>
        public override async Task<T> HandleAsync<T>(Func<Task<T>> next, CancellationToken ct = default)
        {
            return await _policy.ExecuteAsync(async () =>
            {
                return await base.HandleAsync(next, ct).ConfigureAwait(false);
            }).ConfigureAwait(false);
        }
    }
}
