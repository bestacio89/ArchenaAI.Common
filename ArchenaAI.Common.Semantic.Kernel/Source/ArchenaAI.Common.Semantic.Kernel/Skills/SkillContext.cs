using System;
using System.Collections.Generic;

namespace ArchenaAI.Common.Semantic.Kernel.Skills
{
    /// <summary>
    /// Represents the execution context of a skill, carrying input,
    /// ambient metadata, correlation identifiers, and typed helper utilities.
    /// </summary>
    public sealed class SkillContext
    {
        /// <summary>
        /// The raw input provided by the caller. Can be any type.
        /// </summary>
        public object? Input { get; set; }

        /// <summary>
        /// Arbitrary metadata, bindings, or shared objects for the skill execution pipeline.
        /// Behaves similarly to HttpContext.Items or ActorContext properties.
        /// </summary>
        public IDictionary<string, object?> Items { get; }

        /// <summary>
        /// Unique ID used for correlation, logging, tracing, and pipeline consistency.
        /// </summary>
        public string CorrelationId { get; }

        // ------------------------------------------------------------
        // Constructors
        // ------------------------------------------------------------

        /// <summary>
        /// Creates an empty skill context.
        /// </summary>
        public SkillContext()
        {
            Items = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
            CorrelationId = Guid.NewGuid().ToString("N");
        }

        /// <summary>
        /// Creates a skill context containing the given input.
        /// </summary>
        public SkillContext(object? input)
            : this()
        {
            Input = input;
        }

        /// <summary>
        /// Creates a typed skill context.
        /// </summary>
        public static SkillContext From<T>(T input)
        {
            return new SkillContext(input);
        }

        // ------------------------------------------------------------
        // Typed Accessors
        // ------------------------------------------------------------

        /// <summary>
        /// Gets the input value casted to the given type, or throws if incompatible.
        /// </summary>
        public T GetInput<T>()
        {
            if (Input is T typed)
                return typed;

            throw new InvalidCastException(
                $"SkillContext.Input is not of type {typeof(T).Name}.");
        }

        /// <summary>
        /// Attempts to get the input as the given type.
        /// </summary>
        public bool TryGetInput<T>(out T? value)
        {
            if (Input is T typed)
            {
                value = typed;
                return true;
            }

            value = default;
            return false;
        }

        // ------------------------------------------------------------
        // Fluent Item Setters
        // ------------------------------------------------------------

        /// <summary>
        /// Adds or updates an item in the context.
        /// </summary>
        public SkillContext WithItem(string key, object? value)
        {
            Items[key] = value;
            return this;
        }

        /// <summary>
        /// Attempts to retrieve a value from the context items.
        /// </summary>
        public bool TryGetItem<T>(string key, out T? value)
        {
            if (Items.TryGetValue(key, out var raw) && raw is T casted)
            {
                value = casted;
                return true;
            }

            value = default;
            return false;
        }

        /// <summary>
        /// Retrieves an item or throws.
        /// </summary>
        public T GetItem<T>(string key)
        {
            if (Items.TryGetValue(key, out var raw) && raw is T casted)
                return casted;

            throw new KeyNotFoundException(
                $"SkillContext item '{key}' was not found or not of type {typeof(T).Name}.");
        }

        // ------------------------------------------------------------
        // Fluent Input Updater
        // ------------------------------------------------------------

        public SkillContext WithInput(object? input)
        {
            Input = input;
            return this;
        }
    }
}
