using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM;
using ArchenaAI.Common.Semantic.Kernel.Memory.Contracts;
using ArchenaAI.Common.Semantic.Kernel.Pipelines;
using ArchenaAI.Common.Semantic.Kernel.Skills;
using ArchenaAI.Common.Semantic.Kernel.Skills.BuiltIns;
using Microsoft.Extensions.DependencyInjection;
using System;
using System.Collections.Generic;
using System.Reflection;

namespace ArchenaAI.Common.Semantic.Kernel.Builders
{
    /// <summary>
    /// Fluent builder used to construct a fully composed ArchenaKernel instance.
    /// This builder wires together:
    ///   - LLM Connector
    ///   - Memory Providers (symbolic + vector)
    ///   - Memory Manager
    ///   - SkillRegistry + Skills
    ///   - Kernel Pipeline Behaviors
    /// </summary>
    public sealed class ArchenaKernelBuilder : IArchenaKernelBuilder
    {
        private readonly IServiceProvider _services;

        private ILLMConnector? _connector;
        private ISymbolicMemoryProvider? _symbolic;
        private IVectorMemoryProvider? _vector;
        private IMemoryManager? _memoryManager;
        private readonly List<IKernelPipelineBehavior> _behaviors = new();
        private readonly List<IArchenaSkill> _skills = new();

        public ArchenaKernelBuilder(IServiceProvider services)
        {
            _services = services
                ?? throw new ArgumentNullException(nameof(services));
        }

        // -------------------------------------------------------
        // CONNECTOR CONFIGURATION
        // -------------------------------------------------------
        public IArchenaKernelBuilder UseConnector<TConnector>()
            where TConnector : class, ILLMConnector
        {
            _connector = ActivatorUtilities.CreateInstance<TConnector>(_services);
            return this;
        }

        public IArchenaKernelBuilder UseConnector(ILLMConnector connector)
        {
            _connector = connector ?? throw new ArgumentNullException(nameof(connector));
            return this;
        }

        // -------------------------------------------------------
        // MEMORY PROVIDERS
        // -------------------------------------------------------
        public IArchenaKernelBuilder UseSymbolicMemory<TProvider>()
            where TProvider : class, ISymbolicMemoryProvider
        {
            _symbolic = ActivatorUtilities.CreateInstance<TProvider>(_services);
            return this;
        }

        public IArchenaKernelBuilder UseVectorMemory<TProvider>()
            where TProvider : class, IVectorMemoryProvider
        {
            _vector = ActivatorUtilities.CreateInstance<TProvider>(_services);
            return this;
        }

        public IArchenaKernelBuilder UseMemoryManager<TManager>()
            where TManager : class, IMemoryManager
        {
            _memoryManager = ActivatorUtilities.CreateInstance<TManager>(_services);
            return this;
        }

        // -------------------------------------------------------
        // SKILLS
        // -------------------------------------------------------
        public IArchenaKernelBuilder RegisterSkill<TSkill>()
            where TSkill : class, IArchenaSkill
        {
            var skill = ActivatorUtilities.CreateInstance<TSkill>(_services);
            _skills.Add(skill);
            return this;
        }

        public IArchenaKernelBuilder RegisterSkill(IArchenaSkill skill)
        {
            _skills.Add(skill ?? throw new ArgumentNullException(nameof(skill)));
            return this;
        }

        public IArchenaKernelBuilder RegisterSkillsFromAssembly(Assembly assembly)
        {
            foreach (var type in assembly.GetTypes())
            {
                if (!typeof(IArchenaSkill).IsAssignableFrom(type))
                    continue;

                if (type.GetCustomAttribute<SkillAttribute>() is null)
                    continue;

                if (ActivatorUtilities.CreateInstance(_services, type) is IArchenaSkill skill)
                    _skills.Add(skill);
            }

            return this;
        }

        public IArchenaKernelBuilder AddDefaultSkills()
        {
            return RegisterSkill<ReasoningSkill>()
                .RegisterSkill<PlanningSkill>()
                .RegisterSkill<SummarizationSkill>()
                .RegisterSkill<ReflectionSkill>()
                .RegisterSkill<CorrectionSkill>();
        }

        // -------------------------------------------------------
        // PIPELINE BEHAVIORS
        // -------------------------------------------------------
        public IArchenaKernelBuilder AddPipelineBehavior<TBehavior>()
            where TBehavior : class, IKernelPipelineBehavior
        {
            var behavior = ActivatorUtilities.CreateInstance<TBehavior>(_services);
            _behaviors.Add(behavior);
            return this;
        }

        // -------------------------------------------------------
        // BUILD KERNEL
        // -------------------------------------------------------
        public IArchenaKernel Build()
        {
            // LLM Connector
            _connector ??= _services.GetRequiredService<ILLMConnector>();

            // Memory Providers
            _symbolic ??= _services.GetRequiredService<ISymbolicMemoryProvider>();
            _vector ??= _services.GetRequiredService<IVectorMemoryProvider>();

            // Memory Manager
            _memoryManager ??= _services.GetRequiredService<IMemoryManager>();

            // Pipeline Executor
            var pipeline = new PipelineExecutor(_behaviors);

            // Skill Registry (pipeline-enforced)
            var registry = new SkillRegistry(pipeline);
            foreach (var s in _skills)
                registry.Register(s);

            // FINAL kernel compose
            return new ArchenaKernel(
                registry,
                pipeline,
                _connector,
                _memoryManager);
        }
    }
}
