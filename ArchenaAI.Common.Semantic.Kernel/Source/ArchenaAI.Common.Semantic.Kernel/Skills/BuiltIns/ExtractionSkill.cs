using System;
using System.Threading;
using System.Threading.Tasks;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models;
using ArchenaAI.Common.Semantic.Kernel.Skills;

namespace ArchenaAI.Common.Semantic.Kernel.Skills.BuiltIns
{
    [Skill(
        "extraction",
        Description = "Extracts structured data (JSON, key-value, YAML, or bullet points) from unstructured text.",
        Category = SkillCategory.Utility,
        Tags = new[] { "extract", "json", "structure" },
        InputType = typeof(string),
        OutputType = typeof(string)
    )]
    public sealed class ExtractionSkill : IArchenaSkill
    {
        private readonly ILLMConnector _llm;

        public ExtractionSkill(ILLMConnector llm)
        {
            _llm = llm ?? throw new ArgumentNullException(nameof(llm));

            Descriptor = new SkillDescriptor(
                name: Name,
                description: "Extracts structured information (JSON/YAML) from text.",
                category: SkillCategory.Utility,
                tags: new[] { "extraction", "structure" },
                inputType: typeof(string),
                outputType: typeof(string)
            );
        }

        public string Name => "extraction";

        public SkillDescriptor Descriptor { get; }

        public async Task<SkillResult> ExecuteAsync(SkillContext context, CancellationToken cancellationToken = default)
        {
            if (context.Input is not string text || string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("ExtractionSkill requires a non-empty string.");

            var request = new LLMRequest
            {
                SystemPrompt = "You extract structured JSON or YAML from text. Only output valid JSON by default.",
                Prompt = $"Extract structured information from the following text and return as JSON:\n\n{text}"
            };

            request.Metadata["correlationId"] = context.CorrelationId;

            var result = await _llm.SendAsync(request, cancellationToken);
            return SkillResult.FromSuccess(result?.Content ?? "{}");
        }
    }
}
