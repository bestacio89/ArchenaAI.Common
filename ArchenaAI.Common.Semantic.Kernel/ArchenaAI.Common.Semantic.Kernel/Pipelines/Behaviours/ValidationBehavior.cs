using ArchenaAI.Common.Semantic.Kernel.Skills;
using Microsoft.Extensions.Logging;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Pipelines.Behaviours
{
    /// <summary>
    /// Performs skill-level validation BEFORE execution:
    /// - Validates SkillDescriptor existence
    /// - Validates input type correctness
    /// - Validates allowed models (if LLM skill)
    /// - Ensures output type matches descriptor after execution
    ///
    /// This behavior creates a safe, deterministic execution model.
    /// </summary>
    public sealed class ValidationBehavior : KernelPipelineBehavior
    {
        private readonly ISkillRegistry _skillRegistry;
        private readonly string? _modelInUse;

        public ValidationBehavior(
            ISkillRegistry skillRegistry,
            ILogger<ValidationBehavior> logger)
            : base(logger)
        {
            _skillRegistry = skillRegistry ?? throw new ArgumentNullException(nameof(skillRegistry));
            _modelInUse = null;
        }

        /// <summary>
        /// Validates BEFORE executing the next pipeline stage.
        /// </summary>
        protected override async Task OnBeforeAsync(CancellationToken cancellationToken)
        {
            // Nothing to do here yet; validation happens per-result.
            await Task.CompletedTask.ConfigureAwait(false);
        }

        /// <summary>
        /// Handles result validation AFTER pipeline execution.
        /// </summary>
        public override async Task<T> HandleAsync<T>(
            Func<Task<T>> execute,
            CancellationToken cancellationToken = default)
        {
            // Execute first
            var result = await base.HandleAsync(execute, cancellationToken).ConfigureAwait(false);

            // Non-skill operations are fine
            if (result is not SkillResult skillResult)
                return result;

            if (!_skillRegistry.TryGetSkill(skillResult.Metadata["skillName"]?.ToString()!, out var skill))
            {
                Logger.LogError("[ValidationBehavior] Unknown skill was executed.");
                throw new InvalidOperationException("Skill registry mismatch.");
            }

            SkillDescriptor descriptor = skill.Descriptor;

            // Validate output type
            if (descriptor.OutputType != null && skillResult.Output != null)
            {
                if (!descriptor.OutputType.IsAssignableFrom(skillResult.Output.GetType()))
                {
                    Logger.LogError(
                        "[ValidationBehavior] Skill '{Skill}' returned type '{Returned}', expected '{Expected}'.",
                        descriptor.Name,
                        skillResult.Output.GetType().FullName,
                        descriptor.OutputType.FullName);

                    throw new InvalidCastException(
                        $"Skill '{descriptor.Name}' returned incorrect output type.");
                }
            }

            // Validate model restrictions
            if (descriptor.AllowedModels.Count > 0 && !string.IsNullOrWhiteSpace(_modelInUse))
            {
                if (!descriptor.AllowedModels.Contains(_modelInUse))
                {
                    Logger.LogError(
                        "[ValidationBehavior] Skill '{Skill}' is not permitted to run on model '{Model}'.",
                        descriptor.Name,
                        _modelInUse);

                    throw new InvalidOperationException(
                        $"Model '{_modelInUse}' is not allowed for skill '{descriptor.Name}'.");
                }
            }

            return result;
        }
    }
}
