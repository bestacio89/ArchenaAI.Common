using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM;
using ArchenaAI.Common.Semantic.Kernel.Memory.Contracts;
using ArchenaAI.Common.Semantic.Kernel.Pipelines;
using ArchenaAI.Common.Semantic.Kernel.Skills;
using System.Reflection;

namespace ArchenaAI.Common.Semantic.Kernel.Builders
{
    public interface IArchenaKernelBuilder
    {
        // LLM Connector
        IArchenaKernelBuilder UseConnector<TConnector>()
            where TConnector : class, ILLMConnector;

        IArchenaKernelBuilder UseConnector(ILLMConnector connector);

        // Memory Providers
        IArchenaKernelBuilder UseSymbolicMemory<TProvider>()
            where TProvider : class, ISymbolicMemoryProvider;

        IArchenaKernelBuilder UseVectorMemory<TProvider>()
            where TProvider : class, IVectorMemoryProvider;

        IArchenaKernelBuilder UseMemoryManager<TManager>()
            where TManager : class, IMemoryManager;

        // Skills
        IArchenaKernelBuilder RegisterSkill<TSkill>()
            where TSkill : class, IArchenaSkill;

        IArchenaKernelBuilder RegisterSkill(IArchenaSkill skill);

        IArchenaKernelBuilder RegisterSkillsFromAssembly(Assembly assembly);

        IArchenaKernelBuilder AddDefaultSkills(); // reasoning, planning, summarization, correction, reflection

        // Pipeline Behaviors
        IArchenaKernelBuilder AddPipelineBehavior<TBehavior>()
            where TBehavior : class, IKernelPipelineBehavior;

        // Build
        IArchenaKernel Build();
    }
}
