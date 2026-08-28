# VirtualME x CT_AI System Architecture

```mermaid
flowchart TD
    A[Personal records and scene frames] --> B[Consent and local redaction gateway]
    B --> C[CT_AI temporal trace loader]
    C --> D[Spatial Evidence Adapter]
    D --> E[Object and relation records]
    E --> F[Episode clustering]
    F --> G[Deterministic baseline features]
    G --> H[Pattern candidates]

    H --> I[Evidence critic]
    I --> J{User review}
    J -->|Accept or reinterpret| K[Versioned VirtualME self-model]
    J -->|Reject| L[Append-only correction log]
    L --> K

    K --> M[Spatial Resonance Room]
    K --> N[Wave Biography]
    K --> O[Wave Creature renderer]
    K --> P[Research report]

    Q[Model evaluation] --> G
    Q --> H
    R[Privacy and claim gate] --> I
    R --> K

    classDef evidence fill:#e8f1ed,stroke:#46766a,color:#17231f;
    classDef review fill:#fff2d8,stroke:#b98228,color:#33240d;
    classDef output fill:#e7edf8,stroke:#4e6e9c,color:#172033;
    class D,E,F,G evidence;
    class I,J,L,R review;
    class M,N,O,P,K output;
```

## Ownership boundaries

| Component | Responsibility | Must not do |
|---|---|---|
| Consent gateway | consent, redaction, local routing | invent missing records |
| CT_AI trace loader | validate temporal 3D trace | interpret personal intent |
| Spatial Evidence Adapter | normalize objects, relations, uncertainty | infer emotion or personality |
| Episode clustering | group records from one occasion | count one occasion as many witnesses |
| Baseline features | deterministic aggregate measurements | present features as conclusions |
| Evidence critic | reject unsupported claims | rewrite evidence silently |
| User review | choose or correct readings | alter computed confidence |
| Renderer | render the approved model | add unobserved objects or fill gaps |
| Model evaluation | compare baselines and models | use random splits when they leak time or space |

## Data flow invariant

```mermaid
sequenceDiagram
    participant U as User data
    participant G as Gateway
    participant T as CT_AI trace
    participant A as Adapter
    participant V as VirtualME
    participant R as Review
    participant O as Renderer

    U->>G: Consent-scoped records
    G->>T: Local scene trace
    T->>A: Validated objects, covariances, relations
    A->>V: SpatialEvidenceRecord[]
    V->>R: Candidate patterns with provenance
    R->>V: Accepted, rejected, or reinterpreted reading
    V->>O: One versioned approved self-model
    O-->>U: Spatial memory and design output
```

The adapter is observational. `SpatialEvidenceRecord.evidence_class` is `contextual` until a separate reviewable process establishes a stronger evidence classification.
