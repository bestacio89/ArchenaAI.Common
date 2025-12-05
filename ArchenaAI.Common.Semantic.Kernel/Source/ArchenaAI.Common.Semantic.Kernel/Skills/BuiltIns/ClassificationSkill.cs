using System;
using System.Threading;
using System.Threading.Tasks;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models;
using ArchenaAI.Common.Semantic.Kernel.Skills;

namespace ArchenaAI.Common.Semantic.Kernel.Skills.BuiltIns
{
    [Skill(
        "classification",
        Description = "Classifies input into categories based on content.",
        Category = SkillCategory.Utility,
        Tags = new[] { "classify", "category", "label" },
        InputType = typeof(string),
        OutputType = typeof(string)
    )]
    public sealed class ClassificationSkill : IArchenaSkill
    {
        private readonly ILLMConnector _llm;

        public ClassificationSkill(ILLMConnector llm)
        {
            _llm = llm ?? throw new ArgumentNullException(nameof(llm));

            Descriptor = new SkillDescriptor(
                name: Name,
                description: "Classifies text into semantic categories.",
                category: SkillCategory.Utility,
                tags: new[] { "classify", "categorize" },
                inputType: typeof(string),
                outputType: typeof(string)
            );
        }

        public string Name => "classification";

        public SkillDescriptor Descriptor { get; }

        public async Task<SkillResult> ExecuteAsync(SkillContext context, CancellationToken cancellationToken = default)
        {
            if (context.Input is not string text || string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("ClassificationSkill requires non-empty input.");

            var request = new LLMRequest
            {
                SystemPrompt = "You classify input text precisely. Output a JSON object: { \"category\": \"value\" }",
                Prompt = $"Classify the following text:\n\n{text}"
            };

            request.Metadata["correlationId"] = context.CorrelationId;

            var result = await _llm.SendAsync(request, cancellationToken);
            return SkillResult.FromSuccess(result?.Content ?? "{\"category\":\"unknown\"}");
        }
    }
}
