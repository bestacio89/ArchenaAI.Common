namespace ArchenaAI.Common.Semantic.Kernel.Skills;

public enum SkillCategory
{
    Unknown = 0,
    /// <summary>Skill driven primarily by an LLM / chat completion.</summary>
    Llm = 1,
    /// <summary>Skill that calls local code only (no external model).</summary>
    Local = 2,
    /// <summary>Skill that integrates with HTTP / REST APIs.</summary>
    Http = 3,
    /// <summary>Skill that integrates with messaging / brokers (Kafka, etc.).</summary>
    Event = 4,
    /// <summary>Skill that manipulates or queries memory.</summary>
    Memory = 5,
    /// <summary>Planner / orchestration / meta-level reasoning skill.</summary>
    Planning = 6,
    /// <summary>Infrastructure / diagnostic / support skill.</summary>
    Utility = 7
}
