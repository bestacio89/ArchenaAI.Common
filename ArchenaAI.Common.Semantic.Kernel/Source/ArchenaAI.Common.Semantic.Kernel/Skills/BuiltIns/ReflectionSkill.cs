using System;
using System.Threading;
using System.Threading.Tasks;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models;
using ArchenaAI.Common.Semantic.Kernel.Skills;

namespace ArchenaAI.Common.Semantic.Kernel.Skills.BuiltIns
{
    [Skill(
        "reflection",
        Description = "Analyzes and critiques previous outputs to improve quality.",
        Category = SkillCategory.Utility,
        Tags = new[] { "reflection", "critique", "analysis" },
        InputType = typeof(string),
        OutputType = typeof(string)
    )]
    public sealed class ReflectionSkill : IArchenaSkill
    {
        private readonly ILLMConnector _llm;

        public ReflectionSkill(ILLMConnector llm)
        {
            _llm = llm ?? throw new ArgumentNullException(nameof(llm));

            Descriptor = new SkillDescriptor(
                name: Name,
                description: "Critically analyzes text and provides improved alternatives.",
                category: SkillCategory.Utility,
                tags: new[] { "reflection", "critique" },
                inputType: typeof(string),
                outputType: typeof(string)
            );
        }

        public string Name => "reflection";

        public SkillDescriptor Descriptor { get; }

        public async Task<SkillResult> ExecuteAsync(
            SkillContext context,
            CancellationToken cancellationToken = default)
        {
            if (context.Input is not string text || string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("ReflectionSkill requires non-empty string input.");

            var request = new LLMRequest
            {
                Prompt = $"Reflect on the following text and suggest improvements:\n\n{text}",
                SystemPrompt = "You are a reflective critic. Provide insights, mistakes, improvements, and refined versions.",
            };

            request.Metadata["correlationId"] = context.CorrelationId;

            var response = await _llm.SendAsync(request, cancellationToken);
            var output = response?.Content ?? string.Empty;

            return SkillResult.FromSuccess(output);
        }
    }
}
