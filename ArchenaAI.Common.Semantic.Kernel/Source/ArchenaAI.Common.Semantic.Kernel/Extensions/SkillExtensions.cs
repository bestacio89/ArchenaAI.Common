using ArchenaAI.Common.Semantic.Kernel.Skills;
using ArchenaAI.Common.Semantic.Kernel.Skills.BuiltIns;
using Microsoft.Extensions.DependencyInjection;
using System.Reflection;

namespace ArchenaAI.Common.Semantic.Kernel.Extensions
{
    public static class SkillExtensions
    {
        /// <summary>
        /// Registers all skills marked with [Skill] attributes in the given assembly.
        /// </summary>
        public static IServiceCollection AddSkillsFromAssembly(
            this IServiceCollection services,
            Assembly assembly)
        {
            var registry = new SkillRegistryBootstrapper(services);
            registry.RegisterFromAssembly(assembly);
            return services;
        }

        /// <summary>
        /// Registers a single concrete skill type.
        /// </summary>
        public static IServiceCollection AddSkill<TSkill>(this IServiceCollection services)
            where TSkill : class, IArchenaSkill
        {
            services.AddScoped<IArchenaSkill, TSkill>();
            return services;
        }

        /// <summary>
        /// Adds all default built-in skills.
        /// </summary>
        public static IServiceCollection AddDefaultSkills(this IServiceCollection services)
        {
            services.AddSkill<ReasoningSkill>();
            services.AddSkill<ReflectionSkill>();
            services.AddSkill<CorrectionSkill>();
            services.AddSkill<SummarizationSkill>();
            services.AddSkill<PlanningSkill>();
            return services;
        }
    }
}
