using ArchenaAI.Common.Semantic.Kernel.Memory;
using ArchenaAI.Common.Semantic.Kernel.Memory.Contracts;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace ArchenaAI.Common.Semantic.Kernel.Extensions
{
    /// <summary>
    /// Provides DI extensions for registering semantic and symbolic memory components.
    /// </summary>
    public static class MemoryExtensions
    {
        /// <summary>
        /// Registers all memory-related components required by the kernel.
        /// </summary>
        public static IServiceCollection AddKernelMemory(this IServiceCollection services, IConfiguration configuration)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configuration);

            // 🔹 Vector memory provider (semantic embeddings)
            services.AddHttpClient<IVectorMemoryProvider, VectorMemoryProvider>(client =>
            {
                var endpoint = configuration["Kernel:Memory:VectorEndpoint"] ?? "http://localhost:8000";
                client.BaseAddress = new Uri(endpoint);
            });

            // 🔹 Symbolic memory provider (can later support multiple backends)
            // By default, we register a simple in-memory symbolic provider
            services.AddSingleton<ISymbolicMemoryProvider, InMemorySymbolicMemoryProvider>();

            // 🔹 Unified memory manager
            services.AddScoped<IMemoryManager, MemoryManager>();

            // 🔹 Logging for tracing operations
            services.AddLogging(config =>
            {
                config.AddConsole();
                config.AddDebug();
            });

            return services;
        }

        /// <summary>
        /// Registers a custom symbolic memory provider (e.g., backed by SQL, Mongo, or Redis).
        /// </summary>
        public static IServiceCollection AddSymbolicMemory<TProvider>(this IServiceCollection services)
            where TProvider : class, ISymbolicMemoryProvider
        {
            services.AddSingleton<ISymbolicMemoryProvider, TProvider>();
            return services;
        }

        /// <summary>
        /// Registers a custom vector memory provider (e.g., OpenAI, Local Embeddings, etc.).
        /// </summary>
        public static IServiceCollection AddVectorMemory<TProvider>(this IServiceCollection services)
            where TProvider : class, IVectorMemoryProvider
        {
            services.AddScoped<IVectorMemoryProvider, TProvider>();
            return services;
        }
    }
}
