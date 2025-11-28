using System;
using System.Threading;
using System.Threading.Tasks;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models;
using ArchenaAI.Common.Semantic.Kernel.Skills;

namespace ArchenaAI.Common.Semantic.Kernel.Skills.BuiltIns
{
    [Skill(
        "summarization",
        Description = "Produces a clear, concise summary of text.",
        Category = SkillCategory.Llm,
        Tags = new[] { "summary", "shorten", "text" },
        InputType = typeof(string),
        OutputType = typeof(string)
    )]
    public sealed class SummarizationSkill : IArchenaSkill
    {
        private readonly ILLMConnector _llm;

        public SummarizationSkill(ILLMConnector llm)
        {
            _llm = llm ?? throw new ArgumentNullException(nameof(llm));

            Descriptor = new SkillDescriptor(
                name: Name,
                description: "Summarizes input text with clarity and precision.",
                category: SkillCategory.Llm,
                tags: new[] { "summary", "condense" },
                inputType: typeof(string),
                outputType: typeof(string)
            );
        }

        public string Name => "summarization";

        public SkillDescriptor Descriptor { get; }

        public async Task<SkillResult> ExecuteAsync(
            SkillContext context,
            CancellationToken cancellationToken = default)
        {
            if (context.Input is not string text || string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("SummarizationSkill requires a non-empty string.");

            var request = new LLMRequest
            {
                Prompt = $"Summarize the following text clearly:\n\n{text}",
                SystemPrompt = "You are an expert summarizer. Produce concise, accurate summaries with preserved meaning.",
            };

            request.Metadata["correlationId"] = context.CorrelationId;

            var response = await _llm.SendAsync(request, cancellationToken);
            var output = response?.Content ?? string.Empty;

            return SkillResult.FromSuccess(output);
        }
    }
}
