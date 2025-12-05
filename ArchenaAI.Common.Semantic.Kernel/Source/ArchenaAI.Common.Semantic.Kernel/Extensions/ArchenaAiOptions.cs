using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;

namespace ArchenaAI.Common.Semantic.Kernel.Extensions
{
    public sealed class ArchenaAIOptions
    {
        /// <summary>
        /// Whether all skills in the assembly(ies) should be auto-registered.
        /// </summary>
        public bool AutoRegisterSkills { get; set; } = true;

        /// <summary>
        /// Whether default meta-skills (reasoning, planning, reflection, summarization)
        /// should be registered automatically.
        /// </summary>
        public bool IncludeDefaultSkills { get; set; } = true;

        /// <summary>
        /// Assemblies from which skills should be discovered.
        /// Defaults to the assembly that contains ArchenaKernel.
        /// </summary>
        public IList<Assembly> SkillAssemblies { get; } = new List<Assembly>();

        /// <summary>
        /// LLM model ID (e.g. "gpt-4.1" or "llama3").
        /// </summary>
        public string? DefaultModel { get; set; }

        /// <summary>
        /// API provider type (OpenAI, Ollama, etc.)
        /// </summary>
        public LLMProviderType Provider { get; set; } = LLMProviderType.None;

        /// <summary>
        /// API key for cloud LLM providers.
        /// </summary>
        public string? ApiKey { get; set; }

        public enum LLMProviderType
        {
            None,
            OpenAI,
            Ollama,
            AzureOpenAI
        }
    }
}
