using System;
using System.Threading;
using System.Threading.Tasks;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models;
using ArchenaAI.Common.Semantic.Kernel.Skills;

namespace ArchenaAI.Common.Semantic.Kernel.Skills.BuiltIns
{
    [Skill(
        "comparison",
        Description = "Compares two inputs and highlights differences, strengths, and weaknesses.",
        Category = SkillCategory.Utility,
        Tags = new[] { "compare", "diff", "analysis" },
        InputType = typeof(string),
        OutputType = typeof(string)
    )]
    public sealed class ComparisonSkill : IArchenaSkill
    {
        private readonly ILLMConnector _llm;

        public ComparisonSkill(ILLMConnector llm)
        {
            _llm = llm ?? throw new ArgumentNullException(nameof(llm));

            Descriptor = new SkillDescriptor(
                name: Name,
                description: "Compares text A and B for differences and insights.",
                category: SkillCategory.Utility,
                tags: new[] { "comparison", "diff" },
                inputType: typeof(string),
                outputType: typeof(string)
            );
        }

        public string Name => "comparison";

        public SkillDescriptor Descriptor { get; }

        public async Task<SkillResult> ExecuteAsync(SkillContext context, CancellationToken cancellationToken = default)
        {
            if (context.Input is not string composite || !composite.Contains("||"))
                throw new ArgumentException("ComparisonSkill input must be formatted as 'A || B'.");

            var parts = composite.Split("||", 2, StringSplitOptions.TrimEntries);

            var request = new LLMRequest
            {
                SystemPrompt = "You compare two items objectively and highlight key differences.",
                Prompt = $"Compare the following two texts:\n\nA:\n{parts[0]}\n\nB:\n{parts[1]}"
            };

            request.Metadata["correlationId"] = context.CorrelationId;

            var result = await _llm.SendAsync(request, cancellationToken);
            return SkillResult.FromSuccess(result?.Content ?? string.Empty);
        }
    }
}
