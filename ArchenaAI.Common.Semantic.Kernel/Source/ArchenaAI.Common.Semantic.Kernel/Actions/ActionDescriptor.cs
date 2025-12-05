using System;
using System.Collections.Generic;

namespace ArchenaAI.Common.Semantic.Kernel.Actions
{
    public sealed class ActionDescriptor
    {
        public string Name { get; }
        public string Description { get; }
        public IReadOnlyDictionary<string, string> Parameters { get; }

        public ActionDescriptor(
            string name,
            string description,
            IReadOnlyDictionary<string, string> parameters)
        {
            Name = name;
            Description = description;
            Parameters = parameters;
        }
    }
}
