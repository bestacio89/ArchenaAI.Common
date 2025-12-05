using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using ArchenaAI.Common.Semantic.Kernel.Pipelines;

namespace ArchenaAI.Common.Semantic.Kernel.Skills
{
    public sealed class SkillRegistry : ISkillRegistry
    {
        private readonly ConcurrentDictionary<string, IArchenaSkill> _skills =
            new(StringComparer.OrdinalIgnoreCase);

        private readonly IPipelineExecutor _pipelineExecutor;

        public SkillRegistry(IPipelineExecutor pipelineExecutor)
        {
            _pipelineExecutor = pipelineExecutor
                ?? throw new ArgumentNullException(nameof(pipelineExecutor));
        }

        // -----------------------------------------------------------
        // ISkillRegistry implementation
        // -----------------------------------------------------------

        public IEnumerable<SkillDescriptor> GetAllDescriptors()
        {
            foreach (var skill in _skills.Values)
                yield return skill.Descriptor;
        }

        public bool TryGetDescriptor(string name, out SkillDescriptor? descriptor)
        {
            if (_skills.TryGetValue(name, out var skill))
            {
                descriptor = skill.Descriptor;
                return true;
            }

            descriptor = null;
            return false;
        }

        public bool TryGetSkill(string name, out IArchenaSkill? skill)
        {
            return _skills.TryGetValue(name, out skill);
        }

        public void Register(IArchenaSkill skill)
        {
            if (skill == null)
                throw new ArgumentNullException(nameof(skill));

            _skills[skill.Name] = skill;
        }

        public void RegisterFromAssembly(Assembly assembly)
        {
            if (assembly == null)
                throw new ArgumentNullException(nameof(assembly));

            foreach (var type in assembly.GetTypes())
            {
                var attribute = type.GetCustomAttribute<SkillAttribute>();
                if (attribute == null)
                    continue;

                if (!typeof(IArchenaSkill).IsAssignableFrom(type))
                    continue;

                if (Activator.CreateInstance(type, nonPublic: false) is not IArchenaSkill skill)
                    throw new InvalidOperationException(
                        $"Skill '{type.FullName}' must have a public constructor.");

                Register(skill);
            }
        }

        // -----------------------------------------------------------
        // Pipeline-enforced execution
        // -----------------------------------------------------------

        public async Task<SkillResult> ExecuteAsync(
            string skillName,
            SkillContext context,
            CancellationToken cancellationToken = default)
        {
            if (!_skills.TryGetValue(skillName, out var skill))
                throw new KeyNotFoundException($"Skill '{skillName}' not registered.");

            // ALL skill executions go through Franz pipeline enforcement
            return await _pipelineExecutor.ExecuteAsync(
                executeAsync: () => skill.ExecuteAsync(context, cancellationToken),
                cancellationToken: cancellationToken);
        }
    }
}
