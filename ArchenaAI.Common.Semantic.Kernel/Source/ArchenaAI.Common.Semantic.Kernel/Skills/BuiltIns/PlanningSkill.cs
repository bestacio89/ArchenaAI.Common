using System;
using System.Threading;
using System.Threading.Tasks;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM;
using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM.Models;
using ArchenaAI.Common.Semantic.Kernel.Skills;

namespace ArchenaAI.Common.Semantic.Kernel.Skills.BuiltIns
{
    [Skill(
        "planning",
        Description = "Produces a structured, step-by-step plan for a given task.",
        Category = SkillCategory.Planning,
        Tags = new[] { "plan", "task", "strategy", "steps" },
        InputType = typeof(string),
        OutputType = typeof(string)
    )]
    public sealed class PlanningSkill : IArchenaSkill
    {
        private readonly ILLMConnector _llm;

        public PlanningSkill(ILLMConnector llm)
        {
            _llm = llm ?? throw new ArgumentNullException(nameof(llm));

            Descriptor = new SkillDescriptor(
                name: Name,
                description: "Analyzes goals and creates structured multi-step plans.",
                category: SkillCategory.Planning,
                tags: new[] { "plan", "task", "strategy", "steps" },
                inputType: typeof(string),
                outputType: typeof(string)
            );
        }

        public string Name => "planning";

        public SkillDescriptor Descriptor { get; }

        public async Task<SkillResult> ExecuteAsync(
            SkillContext context,
            CancellationToken cancellationToken = default)
        {
            if (context.Input is not string goal || string.IsNullOrWhiteSpace(goal))
                throw new ArgumentException("PlanningSkill requires a non-empty string.");

            var request = new LLMRequest
            {
                Prompt = $"Task: {goal}\n\nCreate a clear, numbered execution plan with actionable steps.",
                SystemPrompt = "You are a strategic planning assistant. Create structured, logical, and efficient plans.",
            };

            request.Metadata["correlationId"] = context.CorrelationId;

            var response = await _llm.SendAsync(request, cancellationToken);
            var output = response?.Content ?? string.Empty;

            return SkillResult.FromSuccess(output);
        }
    }
}
