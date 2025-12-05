using ArchenaAI.Common.Semantic.Kernel.Actions;
using ArchenaAI.Common.Semantic.Kernel.Planning;
using ArchenaAI.Common.Semantic.Kernel.Pipelines;
using Microsoft.Extensions.DependencyInjection;

namespace ArchenaAI.Common.Semantic.Kernel.Extensions
{
    /// <summary>
    /// DI extensions for wiring Franz-compatible kernel pipeline behaviors.
    /// </summary>
    public static class KernelPipelineExtensions
    {
        /// <summary>
        /// Registers the LLM action pipeline behavior so that any LLM-backed
        /// skill result is inspected for structured action calls and, if found,
        /// routed through the ActionRegistry.
        /// </summary>
        public static IServiceCollection AddKernelActionPipeline(this IServiceCollection services)
        {
            // Planner + behavior
            services.AddScoped<LLMActionPlanner>();
            services.AddScoped<IKernelPipelineBehavior, LLMActionPipelineBehavior>();

            return services;
        }
    }
}
