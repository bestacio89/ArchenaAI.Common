using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Skills
{
    /// <summary>
    /// Represents a deterministic, composable skill that can be executed
    /// by the ArchenaAI semantic kernel.
    /// 
    /// Skills may be:
    /// - Local (pure logic)
    /// - LLM-based
    /// - HTTP / API-based
    /// - Event-driven (Kafka, broker)
    /// - Memory-based
    /// - Planner / meta-skills
    /// </summary>
    public interface IArchenaSkill
    {
        /// <summary>
        /// Unique, stable name of the skill.
        /// Must match <see cref="Descriptor"/>.Name exactly.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Metadata defining the skill for discovery, orchestration,
        /// UI tooling, and planner integration.
        /// </summary>
        SkillDescriptor Descriptor { get; }

        /// <summary>
        /// Executes the skill using the provided skill context.
        /// </summary>
        /// <param name="context">Input, memory access, metadata, and service bag.</param>
        /// <param name="cancellationToken">Ambient cancellation signal.</param>
        /// <returns>A <see cref="SkillResult"/> representing success or failure.</returns>
        Task<SkillResult> ExecuteAsync(
            SkillContext context,
            CancellationToken cancellationToken = default
        );
    }
}
