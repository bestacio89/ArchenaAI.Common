using System;
using System.Threading;
using System.Threading.Tasks;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models;
using ArchenaAI.Common.Semantic.Kernel.Skills;

namespace ArchenaAI.Common.Semantic.Kernel.Skills.BuiltIns
{
    [Skill(
        "reasoning",
        Description = "Performs deep logical reasoning and structured analysis.",
        Category = SkillCategory.Llm,
        Tags = new[] { "logic", "reasoning", "analysis" },
        InputType = typeof(string),
        OutputType = typeof(string)
    )]
    public sealed class ReasoningSkill : IArchenaSkill
    {
        private readonly ILLMConnector _llm;

        public ReasoningSkill(ILLMConnector llm)
        {
            _llm = llm ?? throw new ArgumentNullException(nameof(llm));

            Descriptor = new SkillDescriptor(
                name: Name,
                description: "Deep logical analysis and structured reasoning.",
                category: SkillCategory.Llm,
                tags: new[] { "reasoning", "logic", "analysis" },
                inputType: typeof(string),
                outputType: typeof(string)
            );
        }

        public string Name => "reasoning";

        public SkillDescriptor Descriptor { get; }

        public async Task<SkillResult> ExecuteAsync(
            SkillContext context,
            CancellationToken cancellationToken = default)
        {
            if (context.Input is not string question || string.IsNullOrWhiteSpace(question))
                throw new ArgumentException("ReasoningSkill requires a non-empty string.");

            var request = new LLMRequest
            {
                Prompt = $"Question: {question}\n\nProvide structured, step-by-step reasoning.",
                SystemPrompt = "You are an expert reasoner. Produce rigorous logical arguments with verifiable steps.",
            };

            request.Metadata["correlationId"] = context.CorrelationId;

            var response = await _llm.SendAsync(request, cancellationToken);
            var output = response?.Content ?? string.Empty;

            return SkillResult.FromSuccess(output);
        }
    }
}
