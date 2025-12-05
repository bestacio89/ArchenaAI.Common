using System;
using System.Threading;
using System.Threading.Tasks;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models;
using ArchenaAI.Common.Semantic.Kernel.Skills;

namespace ArchenaAI.Common.Semantic.Kernel.Skills.BuiltIns
{
    [Skill(
        "correction",
        Description = "Generates an improved, corrected version of a flawed input text.",
        Category = SkillCategory.Utility,
        Tags = new[] { "correction", "improve", "rewrite" },
        InputType = typeof(string),
        OutputType = typeof(string)
    )]
    public sealed class CorrectionSkill : IArchenaSkill
    {
        private readonly ILLMConnector _llm;

        public CorrectionSkill(ILLMConnector llm)
        {
            _llm = llm ?? throw new ArgumentNullException(nameof(llm));

            Descriptor = new SkillDescriptor(
                name: Name,
                description: "Corrects or improves the clarity, logic, or quality of a text.",
                category: SkillCategory.Utility,
                tags: new[] { "correct", "rewrite" },
                inputType: typeof(string),
                outputType : typeof(string)
            );
        }

        public string Name => "correction";

        public SkillDescriptor Descriptor { get; }

        public async Task<SkillResult> ExecuteAsync(SkillContext context, CancellationToken cancellationToken = default)
        {
            if (context.Input is not string text || string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("CorrectionSkill requires a non-empty string.");

            var request = new LLMRequest
            {
                SystemPrompt = "You are a corrective engine. Rewrite the text in a clearer, more correct, and more coherent form.",
                Prompt = $"Please improve and correct the following text:\n\n{text}"
            };

            request.Metadata["correlationId"] = context.CorrelationId;

            var result = await _llm.SendAsync(request, cancellationToken);
            return SkillResult.FromSuccess(result?.Content ?? string.Empty);
        }
    }
}
