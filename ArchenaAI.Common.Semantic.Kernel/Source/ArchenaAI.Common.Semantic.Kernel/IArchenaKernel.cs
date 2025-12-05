using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel
{
    public interface IArchenaKernel
    {
        /// <summary>
        /// Performs a high-level reasoning operation using the kernel’s reasoning meta-skill.
        /// </summary>
        Task<string> ThinkAsync(string input, CancellationToken ct = default);

        /// <summary>
        /// Executes a registered skill by name.
        /// </summary>
        Task<string> ExecuteSkillAsync(string skillName, string input, CancellationToken ct = default);
    }
}
