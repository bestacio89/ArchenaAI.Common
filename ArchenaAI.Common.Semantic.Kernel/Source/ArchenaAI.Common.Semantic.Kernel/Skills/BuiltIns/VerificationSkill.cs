using System;
using System.Threading;
using System.Threading.Tasks;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models;
using ArchenaAI.Common.Semantic.Kernel.Skills;

namespace ArchenaAI.Common.Semantic.Kernel.Skills.BuiltIns
{
    [Skill(
        "verification",
        Description = "Verifies correctness, logical coherence, and factual consistency of input content.",
        Category = SkillCategory.Utility,
        Tags = new[] { "verification", "consistency", "fact-check" },
        InputType = typeof(string),
        OutputType = typeof(string)
    )]
    public sealed class VerificationSkill : IArchenaSkill
    {
        private readonly ILLMConnector _llm;

        public VerificationSkill(ILLMConnector llm)
        {
            _llm = llm ?? throw new ArgumentNullException(nameof(llm));

            Descriptor = new SkillDescriptor(
                name: Name,
                description: "Verifies accuracy and logical consistency.",
                category: SkillCategory.Utility,
                tags: new[] { "verify", "check", "logic" },
                inputType: typeof(string),
                outputType: typeof(string)
            );
        }

        public string Name => "verification";

        public SkillDescriptor Descriptor { get; }

        public async Task<SkillResult> ExecuteAsync(SkillContext context, CancellationToken cancellationToken = default)
        {
            if (context.Input is not string text || string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("VerificationSkill requires non-empty input.");

            var request = new LLMRequest
            {
                SystemPrompt = "You are a strict verifier. Check if the content is accurate, logically coherent, and free of contradictions.",
                Prompt = $"Verify the correctness and coherence of the following content:\n\n{text}"
            };

            request.Metadata["correlationId"] = context.CorrelationId;

            var result = await _llm.SendAsync(request, cancellationToken);
            return SkillResult.FromSuccess(result?.Content ?? string.Empty);
        }
    }
}
