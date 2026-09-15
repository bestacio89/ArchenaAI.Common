
# **AArchenaAI.Common**\

*A Modular AI Orchestration & Semantic Computing Foundation for .NET*

ArchenaAI.Common is the **core framework layer** powering the **ArchenaAI ecosystem** — a modular, deterministic, and extensible AI architecture designed for:

* Semantic reasoning
* Orchestration pipelines
* Agent execution
* Multi-connector interoperability
* Memory and vector retrieval
* Observability
* Deterministic and reproducible AI workflows

AArchenaAI is built to provide **Franz-level engineering discipline** for AI systems — with strong boundaries, modular design, and predictable behavior.

---

## 🧱 **Architecture Overview**

AArchenaAI.Common is structured as a *multi-package ecosystem*:

```
AArchenaAI.Common/
│
├── src/
│   ├── ArchenaAI.Common.Semantic.Kernel/      ← Core semantic kernel
│   ├── ArchenaAI.Common.Semantic.Runtime/     ← (Coming soon) Execution engine
│   ├── ArchenaAI.Common.Semantic.Connectors/  ← (Coming soon) OpenAI, Azure, FS...
│   ├── ArchenaAI.Common.Semantic.Memory/      ← (Coming soon) Vector stores, embeddings
│   └── ArchenaAI.Common.Semantic.Agents/      ← (Coming soon) Agent abstractions
│
└── tests/
    ├── ArchenaAI.Common.Kernel.Tests/
    └── (future test projects)
```

This structure allows AArchenaAI to evolve into a **production-grade AI framework** with isolated, reusable modules and clean NuGet packaging.

---

## 🧠 **Core Principles**

### **1. Deterministic AI Behavior**

Inspired by Franz’s architectural rigor, ArchenaAI avoids “black box” effects by ensuring:

* deterministic pipeline execution
* explicit orchestration graphs
* traceable behavior
* predictable failures

### **2. Modular Extensibility**

Each subsystem is isolated and optional:

* Semantic Kernel
* Runtime
* Memory
* Connectors
* Agents

This avoids monolithic designs and encourages clean integrations.

### **3. Strong Boundaries (Clean Architecture)**

Domains are isolated using principles similar to:

* Clean Architecture
* Franz Module Boundaries
* DDD-inspired subsystem separation

No hidden dependencies. No cross-module sprawl.

### **4. Enterprise-Ready Design**

Built with:

* Observability
* Logging and Tracing
* Testability
* Deterministic pipelines
* Multi-model support (OpenAI, Azure, local models, etc.)

---

## ⚙️ **Current Modules**

### ### **🧠 ArchenaAI.Common.Semantic.Kernel (Available Today)**

This is the heart of semantic orchestration in AArchenaAI.

It provides:

* Skill model abstraction
* Pipelines for orchestration
* Messaging primitives
* Context propagation
* Memory hooks
* Semantic orchestrators
* Deterministic execution

Perfect for building:

* custom AI pipelines
* multi-step processing flows
* hybrid agent/skill systems
* semantic transformations
* context-aware processing engines

---

## 🔜 **Upcoming Modules**

### **⚙ ArchenaAI.Common.Semantic.Runtime**

A full execution engine supporting:

* agent loops
* orchestration scheduling
* backpressure
* concurrency rules
* pipelines execution with deterministic ordering

### **🔌 ArchenaAI.Common.Semantic.Connectors**

Extensible connectors for:

* OpenAI
* Azure OpenAI
* Ollama / local LLMs
* HTTP data sources
* File system scraping

### **🧬 ArchenaAI.Common.Semantic.Memory**

Vector/RAG infrastructure:

* embeddings
* vector stores
* retrievers
* multi-source memory
* semantic lookup

### **👤 ArchenaAI.Common.Semantic.Agents**

Provides:

* agent definitions
* behavioral modeling
* event-based agent loops
* pattern-based reasoning support

---

## 📦 **Installation (NuGet)**

*(After the first publish)*

```bash
dotnet add package ArchenaAI.Common.Semantic.Kernel
```

Future modules will be installable the same way:

```bash
dotnet add package ArchenaAI.Common.Semantic.Runtime
dotnet add package ArchenaAI.Common.Semantic.Connectors
dotnet add package ArchenaAI.Common.Semantic.Memory
dotnet add package ArchenaAI.Common.Semantic.Agents
```

---

## 🚀 Example Usage

```csharp
var kernel = new ArchenaKernel();
kernel.RegisterSkill(new MySkill());

var context = new SemanticContext("Input text...");
var result = await kernel.ExecuteAsync(context);

Console.WriteLine(result.Output);
```

---

## 📚 **Folder Structure**

```
src/
  ArchenaAI.Common.Semantic.Kernel/
      Skills/
      Messaging/
      Pipelines/
      Actions/
      Agents/
      Observability/
      Orchestration/
      Memory/
      ...
tests/
  ArchenaAI.Common.Kernel.Tests/
```

---

## 🤝 Contributing

Pull requests are welcome.
Areas open for contribution:

* new connectors
* memory integrations
* orchestration improvements
* agent behaviors
* observability tooling
* documentation

---

## 🗺 Roadmap

* [ ] Semantic Runtime (Execution Engine)
* [ ] Memory subsystem
* [ ] OpenAI/Azure connectors
* [ ] Agent abstractions
* [ ] Plugin architecture
* [ ] Benchmarks suite

---

## 📜 License

MIT License — short, permissive, open-source friendly.

---


