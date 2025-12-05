using System.Reflection;

namespace ArchenaAI.Common.Semantic.Kernel.Skills
{
    public interface ISkillRegistry
    {
        IEnumerable<SkillDescriptor> GetAllDescriptors();
        bool TryGetDescriptor(string name, out SkillDescriptor? descriptor);
        bool TryGetSkill(string name, out IArchenaSkill? skill);

        void Register(IArchenaSkill skill);
        void RegisterFromAssembly(Assembly assembly);

        Task<SkillResult> ExecuteAsync(
            string skillName,
            SkillContext context,
            CancellationToken cancellationToken = default);
    }
}
