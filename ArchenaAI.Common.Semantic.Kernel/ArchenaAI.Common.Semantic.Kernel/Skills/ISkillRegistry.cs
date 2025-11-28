namespace ArchenaAI.Common.Semantic.Kernel.Skills
{
    public interface ISkillRegistry
    {
        void Register(IArchenaSkill skill);
        IArchenaSkill? GetSkillAsync(string name);
        IEnumerable<IArchenaSkill> GetAll();
    }
}
