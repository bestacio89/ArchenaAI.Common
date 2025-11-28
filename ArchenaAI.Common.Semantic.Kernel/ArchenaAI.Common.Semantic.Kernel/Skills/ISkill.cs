using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Skills
{
    public interface IArchenaSkill
    {
        string Name { get; }
        string Description { get; }

        Task<string> ExecuteAsync(string input, CancellationToken ct);
    }
}
