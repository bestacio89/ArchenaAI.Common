using System;
using System.Threading;
using System.Threading.Tasks;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models;
using ArchenaAI.Common.Semantic.Kernel.Skills;

namespace ArchenaAI.Common.Semantic.Kernel.Skills.BuiltIns
{
    [Skill(
        "critique",
        Description = "Evaluates the quality, clarity, correctness, and completeness of a given text.",
        Category = SkillCategory.Utility,
        Tags = new[] { "critique", "evaluation", "quality-check" },
        InputType = typeof(string),
        OutputType = typeof(string)
    )]
    public sealed class CritiqueSkill : IArchenaSkill
    {
        private readonly ILLMConnector _llm;

        public CritiqueSkill(ILLMConnector llm)
        {
            _llm = llm ?? throw new ArgumentNullException(nameof(llm));

            Descriptor = new SkillDescriptor(
                name: Name,
                description: "Evaluates text for quality, correctness and completeness.",
                category: SkillCategory.Utility,
                tags: new[] { "critique", "evaluation" },
                inputType: typeof(string),
                outputType: typeof(string)
            );
        }

        public string Name => "critique";

        public SkillDescriptor Descriptor { get; }

        public async Task<SkillResult> ExecuteAsync(SkillContext context, CancellationToken cancellationToken = default)
        {
            if (context.Input is not string text || string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("CritiqueSkill requires a non-empty string.");

            var request = new LLMRequest
            {
                SystemPrompt = "You are an expert evaluator. Provide an objective critique of the text.",
                Prompt = $"Please critique the following text:\n\n{text}"
            };

            request.Metadata["correlationId"] = context.CorrelationId;

            var result = await _llm.SendAsync(request, cancellationToken);
            return SkillResult.FromSuccess(result?.Content ?? string.Empty);
        }
    }
}
