using ArchenaAI.Common.Semantic.Kernel.Builders;
using ArchenaAI.Common.Semantic.Kernel.Extensions;
using ArchenaAI.Common.Semantic.Kernel.Orchestration;
using ArchenaAI.Common.Semantic.Kernel.Skills;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace ArchenaAI.Common.Semantic.Kernel.Extensions
{
    public static class ServiceCollectionExtensions
    {
        /// <summary>
        /// Registers the entire ArchenaAI Semantic Kernel stack:
        /// - LLM connectors
        /// - Memory providers
        /// - Franz pipeline behaviors
        /// - SK pipeline behaviors
        /// - Skill registry + default skills
        /// - Kernel + orchestrator
        /// </summary>
        public static IServiceCollection AddArchenaKernel(
            this IServiceCollection services,
            IConfiguration configuration)
        
        {
            // Kernel options from configuration
            services.Configure<LoopOptions>(configuration.GetSection("Kernel:LoopOptions"));
            // Franz pipeline
            services.AddSemanticKernelPipeline();

            // Kernel action pipeline
            services.AddKernelActionPipeline();

            // Memory stack (requires configuration)
            services.AddKernelMemory(configuration);

            // Skill registry
            services.AddSingleton<ISkillRegistry, SkillRegistry>();

            // Default skills
            services.AddDefaultSkills();

            // Kernel façade
            services.AddSingleton<IArchenaKernel, ArchenaKernel>();

            // Orchestrator
            services.AddScoped<ArchenaKernelOrchestrator>();

            return services;
        }

    }
}
