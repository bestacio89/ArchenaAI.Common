using ArchenaAI.Common.MCP.Abstractions;
using ArchenaAI.Common.MCP.Policies;
using ArchenaAI.Common.MCP.Tribunals;
using Microsoft.Extensions.DependencyInjection;

namespace ArchenaAI.Common.MCP.Extensions
{
    public static class ServiceCollectionExtensions
    {
        /// <summary>
        /// Registers the MCP governance layer: policy evaluators + the tribunal
        /// that runs them. Call after AddArchenaKernel().
        /// </summary>
        public static IServiceCollection AddArchenaMcp(this IServiceCollection services)
        {
            services.AddSingleton<IPolicyEvaluator, CapabilityAuthorizationEvaluator>();
            services.AddSingleton<IPolicyEvaluator, BoundaryComplianceEvaluator>();
            services.AddSingleton<RuntimeTribunal>();

            return services;
        }
    }
}