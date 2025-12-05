# 🧠 **ArchenaAI Semantic Kernel**

### *Deterministic, Pipeline-Driven, Skill-Based AI Orchestration Framework*

---

## ⭐ Overview

**ArchenaAI Semantic Kernel** is a modular, deterministic AI orchestration engine designed for:

* Multi-step reasoning
* Tool calling / Action execution
* Memory-augmented loops
* Semantic pipelines (Franz-style behaviors)
* Distributed eventing
* Agent architectures
* LLM + non-LLM hybrid reasoning
* Enterprise-grade reliability
* Kafka-based multi-agent messaging

It is the **foundational SDK** used by the upcoming
**ArchenaAI Runtime**, enabling distributed LLM agents to collaborate using structured messages, skills, tools, and reasoning pipelines.

---

# 📦 Features

### ✔ Skill system

Typed skills with descriptors, metadata, categories, automatic discovery, and deterministic execution.

### ✔ Pipeline execution engine

A Franz-inspired reversible pipeline chain enabling:

* Logging
* Resilience (retry, timeout, breaker)
* Validation
* Memory augmentation
* LLM Action / tool-calling interception

### ✔ Reasoning Orchestrator

Implements ReAct-style loops with:

* Reflection
* Correction
* Planning
* Tool execution
* Termination conditions
* Memory integration
* Semantic event broadcasting

### ✔ Memory stack

* Vector memory (via provider abstraction)
* Symbolic memory
* Unified memory manager
* Context-aware retrieval and injection into reasoning loops

### ✔ Action / Tool System

Strongly typed actions with:

* ActionDescriptor
* ActionContext
* ActionRegistry
* Automated execution from LLM responses

### ✔ Multi-agent ready

Kafka connectors + SemanticOrchestrator allow distributed reasoning across agent clusters.

---

# 🧱 Architecture Overview

Below are diagrams explaining the core architecture.

---

# **1. High-Level Kernel Architecture**

```mermaid
flowchart TD

    subgraph Skills[Skills Layer]
        S1[Skill: Reasoning]
        S2[Skill: Reflection]
        S3[Skill: Correction]
        S4[Skill: Planning]
        S5[Skill: Summarization]
        S6[User-defined Skills]
    end

    subgraph Pipelines[Pipeline Layer]
        P1[LoggingBehavior]
        P2[ResilienceBehavior]
        P3[ValidationBehavior]
        P4[SymbolicMemoryBehavior]
        P5[VectorMemoryBehavior]
        P6[LLMActionPipelineBehavior]
    end

    subgraph Kernel[ArchenaKernel]
        K1[SkillContext]
        K2[SkillRegistry]
        K3[ActionRegistry]
        K4[PipelineExecutor]
        K5[LLM Connector]
        K6[MemoryManager]
    end

    subgraph Orchestrator[ArchenaKernelOrchestrator]
        O1[Reasoning Loop]
        O2[Planning Loop]
        O3[Reflection]
        O4[Tool Execution]
        O5[Event Emission]
    end

    UserInput --> Orchestrator --> Kernel --> Pipelines --> Skills --> Kernel --> Orchestrator --> Output
```

---

# **2. Skill Execution Flow**

```mermaid
sequenceDiagram
    participant User
    participant Orchestrator
    participant Kernel
    participant Pipeline
    participant Skill
    participant Memory

    User->>Orchestrator: Input
    Orchestrator->>Kernel: ExecuteSkill("reasoning")
    Kernel->>Pipeline: Build chain
    Pipeline->>Memory: Retrieve context
    Memory-->>Pipeline: Relevant memories
    Pipeline->>Skill: Invoke skill
    Skill-->>Pipeline: SkillResult
    Pipeline-->>Kernel: SkillResult
    Kernel-->>Orchestrator: Output text
    Orchestrator-->>User: Final answer
```

---

# **3. Pipeline Behavior Chain**

```mermaid
flowchart LR

    A[Input] --> B[LoggingBehavior]
    B --> C[ResilienceBehavior]
    C --> D[ValidationBehavior]
    D --> E[SymbolicMemoryBehavior]
    E --> F[VectorMemoryBehavior]
    F --> G[LLMActionPipelineBehavior]
    G --> H[Skill Execution]
    H --> Z[Output]
```

The pipeline is **reversible**, meaning behaviors wrap the skill call like middleware:

```
Logging
  Resilience
    Validation
      SymbolicMemory
        VectorMemory
          ActionPlanner
             [Skill]
```

---

# **4. Memory Architecture**

```mermaid
flowchart TD

    subgraph SymbolicMemory
        SM1[InMemory Provider]
        SM2[Custom Provider]
    end

    subgraph VectorMemory
        VM1[HTTP Embedding Provider]
        VM2[Custom Vector DB]
    end

    SM1 --> MM[MemoryManager]
    SM2 --> MM
    VM1 --> MM
    VM2 --> MM

    MM --> O[Orchestrator]
    O --> MM
```

---

# **5. Action / Tool Execution**

```mermaid
flowchart TD

    LLM[LLM Output<br/>with tool call JSON]
        --> Planner[LLMActionPlanner<br/>Parse + Validate action]
        --> Reg[ActionRegistry]
        --> Exec[Action.ExecuteAsync<br/>with ActionContext]
        --> Return[Return tool result<br/>as reasoning output]
```

The kernel intercepts tool calls automatically from the LLM text output.

---

# **6. Distributed Semantic Event Bus**

```mermaid
flowchart LR

    Orchestrator --> Serializer
    Serializer --> KafkaConnector
    KafkaConnector --> Topic[archenaai.semantic.events]
```

Agents and external services can subscribe to this topic to observe:

* skill usage
* reasoning state
* meta-cognition
* memory updates
* tool invocations

---

# 🧬 Project Structure

```
ArchenaAI.Semantic.Kernel/
│
├── Skills/
│   ├── SkillAttribute.cs
│   ├── SkillDescriptor.cs
│   ├── SkillContext.cs
│   ├── SkillResult.cs
│   └── BuiltIns/
│
├── Pipelines/
│   ├── PipelineExecutor.cs
│   ├── KernelPipelineBehavior.cs
│   └── Behaviours/
│       ├── LoggingBehavior.cs
│       ├── ResilienceBehavior.cs
│       ├── ValidationBehavior.cs
│       ├── RetryBehavior.cs
│       ├── TimeoutBehavior.cs
│       ├── SymbolicMemoryBehavior.cs
│       └── VectorMemoryBehavior.cs
│
├── Actions/
│   ├── ActionRegistry.cs
│   ├── ActionDescriptor.cs
│   └── ActionContext.cs
│
├── Memory/
│   ├── IVectorMemoryProvider.cs
│   ├── ISymbolicMemoryProvider.cs
│   ├── MemoryManager.cs
│   └── Providers/
│
├── Connectors/
│   ├── LLM/
│   ├── Kafka/
│   └── HTTP/
│
├── Orchestration/
│   └── ArchenaKernelOrchestrator.cs
│
├── Messaging/
│   └── SemanticMessage.cs
│
├── ArchenaKernel.cs
├── IArchenaKernel.cs
└── Extensions/
    └── DependencyInjection Extensions
```

---

# 💻 Quick Start

## Install (when published to NuGet):

```bash
dotnet add package ArchenaAI.Semantic.Kernel
```

---

## 1. Register the kernel

```csharp
services.AddArchenaKernel(configuration);
```

---

## 2. Add skills

```csharp
[Skill("greet", Description = "Simple greeting skill", Category = SkillCategory.Local)]
public sealed class GreetingSkill : IArchenaSkill
{
    public SkillDescriptor Descriptor =>
        new("greet", "Greeting skill", SkillCategory.Local);

    public Task<SkillResult> ExecuteAsync(SkillContext context, CancellationToken ct)
    {
        var name = context.Input?.ToString() ?? "unknown";
        return Task.FromResult(SkillResult.FromSuccess($"Hello {name}!"));
    }
}
```

---

## 3. Run a skill

```csharp
var output = await kernel.ExecuteSkillAsync("greet", "Bernardo");
```

---

## 4. Run high-level reasoning

```csharp
var answer = await kernel.ThinkAsync("Explain microservices in simple terms.");
```

---

# 📌 Design Principles

### 🟩 Deterministic

Explicit skill metadata, typed outputs, validation.

### 🟩 Extensible

Every layer (skills, memory, pipelines, actions) supports custom implementations.

### 🟩 Distributed

Built-in Kafka for multi-agent communication.

### 🟩 Resilient

Retry, timeout, breaker, and validation behaviors.

### 🟩 Hybrid

Supports LLM + classical logic + external tools.

### 🟩 Agentic

Designed to support ArchenaAI Runtime for multi-agent systems.

---

# 🏁 Status

### ✔ **Kernel: 100% Complete**

### ✔ **Production-ready**

### ✔ **Runtime-ready**

### ⏳ Next step: **ArchenaAI Runtime (separate repository)**

---

# 🔮 Roadmap

* ArchenaAI.Runtime (agent host & supervisor)
* Multi-agent collaboration policies
* Agent federation layer
* Aegis security/governance integration
* Visual debugging environment
* MCP (Model Context Protocol) tool servers

---

# ❤️ License

MIT — Open, extensible, developer-friendly.


