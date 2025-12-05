using Microsoft.Extensions.DependencyInjection;
using ArchenaAI.Common.Semantic.Kernel.Pipelines;
using ArchenaAI.Common.Semantic.Kernel.Pipelines.Behaviours;

namespace ArchenaAI.Common.Semantic.Kernel.Extensions
{
    public static class PipelineExtensions
    {
        /// <summary>
        /// Registers all default Semantic Kernel pipeline behaviors.
        /// Order is important: first registered = outermost behavior.
        /// </summary>
        public static IServiceCollection AddSemanticKernelPipeline(this IServiceCollection services)
        {
            // Core behaviors
            services.AddSingleton<IKernelPipelineBehavior, LoggingBehavior>();
            services.AddSingleton<IKernelPipelineBehavior, ResilienceBehavior>();

            // Memory behaviors
            services.AddSingleton<IKernelPipelineBehavior, SymbolicMemoryKernelBehavior>();
            services.AddSingleton<IKernelPipelineBehavior, VectorMemoryKernelBehavior>();

            // Pipeline executor
            services.AddSingleton<IPipelineExecutor, PipelineExecutor>();


            services.AddSingleton<IKernelPipelineBehavior, ValidationBehavior>();
            services.AddSingleton<IKernelPipelineBehavior, RetryBehavior>();
            services.AddSingleton<IKernelPipelineBehavior, TimeoutBehavior>();


            return services;
        }
    }
}
