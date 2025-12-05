using ArchenaAI.Common.Semantic.Kernel.Connectors.LLM;
using ArchenaAI.Common.Semantic.Kernel.Memory.Contracts;
using ArchenaAI.Common.Semantic.Kernel.Pipelines;
using ArchenaAI.Common.Semantic.Kernel.Skills;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel
{
    /// <summary>
    /// The main façade for executing reasoning operations and skills within
    /// the ArchenaAI Semantic Kernel. This class enforces deterministic
    /// execution via the SkillRegistry and underlying pipeline executor.
    /// </summary>
    public sealed class ArchenaKernel : IArchenaKernel
    {
        private readonly ISkillRegistry _skillRegistry;
        private readonly IPipelineExecutor _pipeline;
        private readonly ILLMConnector _connector;
        private readonly IMemoryManager _memory;

        public ArchenaKernel(
            ISkillRegistry skills,
            IPipelineExecutor pipeline,
            ILLMConnector connector,
            IMemoryManager memory)
        {
            _skillRegistry = skills;
            _pipeline = pipeline;
            _connector = connector;
            _memory = memory;
        }

        // --------------------------------------------------------------------
        // THINK: High-level LLM reasoning (uses "reasoning" meta skill)
        // --------------------------------------------------------------------
        public async Task<string> ThinkAsync(string input, CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(input))
                throw new ArgumentException("ThinkAsync requires non-empty input.", nameof(input));

            var context = SkillContext.From(input);

            var result = await _skillRegistry.ExecuteAsync("reasoning", context, ct);

            return result.Output?.ToString() ?? string.Empty;
        }

        // --------------------------------------------------------------------
        // Execute ANY skill by name with string input
        // --------------------------------------------------------------------
        public async Task<string> ExecuteSkillAsync(
            string skillName,
            string input,
            CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(skillName))
                throw new ArgumentException("Skill name cannot be empty.", nameof(skillName));

            var context = SkillContext.From(input);

            var result = await _skillRegistry.ExecuteAsync(skillName, context, ct);

            return result.Output?.ToString() ?? string.Empty;
        }

        // --------------------------------------------------------------------
        // Advanced Execution: Execute ANY skill with ANY typed input
        // --------------------------------------------------------------------
        public async Task<string> ExecuteSkillAsync(
            string skillName,
            object? input,
            CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(skillName))
                throw new ArgumentException("Skill name cannot be empty.", nameof(skillName));

            var context = SkillContext.From(input);

            var result = await _skillRegistry.ExecuteAsync(skillName, context, ct);

            return result.Output?.ToString() ?? string.Empty;
        }

        // --------------------------------------------------------------------
        // Strongly-typed execution (generic)
        // Future-proof for JSON inputs, DTOs, domain objects, etc.
        // --------------------------------------------------------------------
        public async Task<TOut?> ExecuteSkillAsync<TIn, TOut>(
            string skillName,
            TIn input,
            CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(skillName))
                throw new ArgumentException("Skill name cannot be empty.", nameof(skillName));

            var context = SkillContext.From(input);

            var result = await _skillRegistry.ExecuteAsync(skillName, context, ct);

            if (result.Output is TOut casted)
                return casted;

            // LLM skills return string: we can convert if possible
            if (result.Output is string raw && typeof(TOut) == typeof(string))
                return (TOut?)(object)raw;

            // Could extend with JSON deserialization later
            return default;
        }
    }
}
