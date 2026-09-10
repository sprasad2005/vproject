# Software and System Design: UML Diagrams

## Overview
This document presents the formal software engineering and system architecture diagrams for the **RiceGuard** platform. The diagrams model the actual, verified software components implemented across the repository, encompassing the client user interface (`app/frontend/`), asynchronous backend API server (`app/backend/main.py`), core neural inference engines (`src/models/`, `src/inference/`), and data management pipelines (`src/data/`).

---

## 1. System Architecture Diagram
The RiceGuard platform follows a modular, decoupled client-server architecture. The presentation tier communicates with the inference service via asynchronous RESTful HTTP endpoints.

```mermaid
flowchart TB
    subgraph Client_Layer["Presentation Tier (Client Browser)"]
        UI["React 19 + Vite 6 Single Page Application"]
        UploadComp["Image Upload & Drag-and-Drop Handler"]
        ViewerComp["Dual Diagnostic Canvas & Box Overlay Renderer"]
        ParamComp["Interactive Threshold Controls (tau, Top-K)"]
        StateStore["Client State & Error Boundary Manager"]
        
        UI --> UploadComp
        UI --> ViewerComp
        UI --> ParamComp
        UI --> StateStore
    end

    subgraph Network_Boundary["Network & Communication Boundary"]
        HTTPReq["HTTP POST /predict (Multipart Form Data)"]
        HTTPResp["JSON Response Payload (Classes, Boxes, Timings)"]
    end

    subgraph Backend_Layer["Application Service Tier (FastAPI Service)"]
        APIServer["FastAPI / Uvicorn Asynchronous Server (Port 8000)"]
        CORSMiddleware["CORS & Request Validation Middleware"]
        RouteHandler["/predict & /health Route Handlers"]
        PayloadParser["Image Stream & Form Parameter Parser"]
        
        APIServer --> CORSMiddleware
        CORSMiddleware --> RouteHandler
        RouteHandler --> PayloadParser
    end

    subgraph Inference_Engine["Core AI & Inference Engine Tier"]
        PreprocModule["PIL / Torchvision Preprocessor (224x224, Norm)"]
        ModelRuntime["PyTorch 2.6.0 Runtime (CUDA 12.4 / CPU)"]
        ModelWeights["Phase 3B Multi-Task Checkpoint (model_phase3b.pt)"]
        DualHeads["Bifurcated Heads: 6-Class Softmax + 7x7 Grid Loc"]
        CalibrationDecoder["Phase 3B Calibration Decoder (tau=0.60, K=3)"]
        
        PayloadParser --> PreprocModule
        PreprocModule --> ModelRuntime
        ModelWeights --> ModelRuntime
        ModelRuntime --> DualHeads
        DualHeads --> CalibrationDecoder
    end

    Client_Layer <--> Network_Boundary
    Network_Boundary <--> Backend_Layer
    Backend_Layer <--> Inference_Engine

    style Client_Layer fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#ffffff
    style Network_Boundary fill:#1e293b,stroke:#64748b,stroke-width:1px,color:#ffffff
    style Backend_Layer fill:#0f172a,stroke:#10b981,stroke-width:2px,color:#ffffff
    style Inference_Engine fill:#0f172a,stroke:#f59e0b,stroke-width:2px,color:#ffffff
```

---

## 2. Sequence Diagram: Diagnostic Inference Lifecycle
This sequence diagram models the chronological execution flow from the moment an agricultural user uploads a rice leaf image to the final rendering of the calibrated disease diagnosis.

```mermaid
sequenceDiagram
    autonumber
    actor User as Field Inspector / User
    participant Frontend as React 19 Frontend (App.jsx)
    participant API as FastAPI Backend (main.py)
    participant Preproc as Torchvision Preprocessor
    participant Model as Multi-Task Model (EfficientNet-B0)
    participant Decoder as Calibrated Decoder (Phase 3B)

    User->>Frontend: Selects / Drops Rice Leaf Image (JPEG/PNG)
    Frontend->>Frontend: Validates MIME type, file size (<10MB), creates local preview
    Frontend->>API: HTTP POST /predict (image_file, tau=0.60, top_k=3)
    
    activate API
    API->>API: Validates multipart payload and parameters
    API->>Preproc: Pass raw image bytes
    
    activate Preproc
    Preproc->>Preproc: PIL decode -> Resize(224, 224) -> ToTensor() -> Normalize()
    Preproc-->>API: Transformed Tensor [1, 3, 224, 224] on Target Device (CUDA/CPU)
    deactivate Preproc
    
    API->>Model: Forward pass torch.no_grad()
    activate Model
    Model->>Model: Shared Backbone features.8 -> [1, 1280, 7, 7]
    Model->>Model: Cls Head: AdaptiveAvgPool -> Linear -> Logits [1, 6]
    Model->>Model: Loc Head: Conv2d -> SiLU -> Conv2d -> Grid [1, 5, 7, 7]
    Model-->>API: Raw Tensors: (cls_logits, loc_grid)
    deactivate Model

    API->>Decoder: Decode & Calibrate (cls_logits, loc_grid, tau, top_k)
    activate Decoder
    Decoder->>Decoder: Apply Softmax -> Top-1 Class & Confidence
    Decoder->>Decoder: Sigmoid Objectness -> Filter (score >= tau)
    Decoder->>Decoder: Decode (bx, by, bw, bh) -> Sort -> Truncate to Top-K
    Decoder-->>API: Structured Result (class_name, confidence, boxes[], latency_ms)
    deactivate Decoder

    API-->>Frontend: HTTP 200 OK (JSON Diagnostic Payload)
    deactivate API

    Frontend->>Frontend: Parse JSON payload, compute viewport scaling
    Frontend->>Frontend: Render diagnosis badge, confidence bars, and canvas bounding boxes
    Frontend-->>User: Display Dual-Pane Diagnostic View with Calibrated Overlays
```

---

## 3. Component Diagram: Software Subsystem Decomposition
The component diagram outlines the internal structural dependencies between the Python backend modules and the React frontend components.

```mermaid
flowchart LR
    subgraph Frontend_Components["React Frontend Subsystem"]
        AppUI["App.jsx<br/>(Root Container)"]
        DropzoneComp["ImageUploader.jsx<br/>(Drag-and-Drop Handler)"]
        DiagnosisCard["DiagnosisCard.jsx<br/>(Class & Confidence Display)"]
        CanvasOverlay["CanvasOverlay.jsx<br/>(Bounding Box Renderer)"]
        ConfigBar["ConfigControls.jsx<br/>(Threshold Sliders)"]

        AppUI --> DropzoneComp
        AppUI --> DiagnosisCard
        AppUI --> CanvasOverlay
        AppUI --> ConfigBar
    end

    subgraph Backend_Components["FastAPI Backend Subsystem"]
        APIRouter["main.py<br/>(FastAPI Application)"]
        InferService["predict_service.py<br/>(Inference Pipeline)"]
        ModelLoader["model_loader.py<br/>(Checkpoint Manager)"]
        
        APIRouter --> InferService
        InferService --> ModelLoader
    end

    subgraph Core_ML_Package["Core PyTorch Library (src/)"]
        NetArch["src.models.multitask_efficientnet<br/>(MultiTaskEfficientNetB0)"]
        Transforms["src.data.transforms<br/>(Preprocessing Pipelines)"]
        PostProc["src.inference.decoding<br/>(CalibratedBoxDecoder)"]
        XAIModule["src.xai.gradcam<br/>(GradCAMGrounding)"]

        ModelLoader --> NetArch
        InferService --> Transforms
        InferService --> PostProc
        NetArch -.-> XAIModule
    end

    Frontend_Components <==>|"REST / JSON via Axios / Fetch"| Backend_Components

    style Frontend_Components fill:#0f172a,stroke:#3b82f6,stroke-width:1px,color:#ffffff
    style Backend_Components fill:#0f172a,stroke:#10b981,stroke-width:1px,color:#ffffff
    style Core_ML_Package fill:#0f172a,stroke:#f59e0b,stroke-width:1px,color:#ffffff
```

---

## 4. Class Diagram: Core Domain and Architectural Entities
The class diagram details the object-oriented structure of the neural network architectures, data loaders, and calibrated decoders implemented in `src/`.

```mermaid
classDiagram
    class nn_Module {
        +forward(*args)
        +to(device)
        +eval()
        +train()
    }

    class MultiTaskEfficientNetB0 {
        -features: Sequential
        -avgpool: AdaptiveAvgPool2d
        -dropout: Dropout
        -classifier: Linear
        -loc_conv1: Conv2d
        -loc_bn: BatchNorm2d
        -loc_act: SiLU
        -loc_conv2: Conv2d
        +forward(x: Tensor) Tuple~Tensor, Tensor~
        +extract_features(x: Tensor) Tensor
    }

    class BaselineEfficientNetB0 {
        -features: Sequential
        -avgpool: AdaptiveAvgPool2d
        -dropout: Dropout
        -classifier: Linear
        +forward(x: Tensor) Tensor
    }

    class CalibratedBoxDecoder {
        -tau: float
        -top_k: int
        -grid_size: int
        +decode_boxes(loc_tensor: Tensor, tau: float, top_k: int) List~Dict~
        +filter_by_confidence(candidates: List, tau: float) List
        +non_max_suppression(boxes: Tensor, scores: Tensor) Tensor
    }

    class GradCAMGrounding {
        -model: nn_Module
        -target_layer: nn_Module
        -gradients: Tensor
        -activations: Tensor
        +generate_cam(input_tensor: Tensor, target_class: int) ndarray
        +compute_energy_inside_mask(cam: ndarray, mask: ndarray) float
        +compute_attribution_iou(cam: ndarray, mask: ndarray) float
        +compute_pointing_hit(cam: ndarray, mask: ndarray) bool
    }

    class RiceDiseaseDataset {
        -image_paths: List~str~
        -labels: List~int~
        -boxes: List~List~
        -transforms: Callable
        +__len__() int
        +__getitem__(idx: int) Tuple~Tensor, int, Tensor~
    }

    nn_Module <|-- MultiTaskEfficientNetB0
    nn_Module <|-- BaselineEfficientNetB0
    MultiTaskEfficientNetB0 --> CalibratedBoxDecoder : uses
    MultiTaskEfficientNetB0 --> GradCAMGrounding : inspected by
    RiceDiseaseDataset --> MultiTaskEfficientNetB0 : feeds
```

---

## 5. Activity Diagram: Image Ingestion and Dual-Output Diagnostic Flow
The activity diagram captures the decision branches and error-handling mechanisms active during client-side image processing and server-side model execution.

```mermaid
flowchart TD
    Start([User Initiates Diagnosis]) --> SelectImg[Select or Drop Image File]
    SelectImg --> ValClient{Valid Format & Size?}
    
    ValClient -- No --> ShowClientErr[Display Client Validation Alert] --> SelectImg
    ValClient -- Yes --> GenPreview[Render Local UI Thumbnail]
    
    GenPreview --> SendAPI[Send Multipart POST Request to /predict]
    SendAPI --> ValServer{Server Active & Payload Valid?}
    
    ValServer -- No --> ReturnHTTPError[Return HTTP 4xx/5xx Error]
    ReturnHTTPError --> ShowNetworkErr[Display User-Friendly Error Toast]
    
    ValServer -- Yes --> DecodeImg[Decode PIL Image & Convert RGB]
    DecodeImg --> ApplyTransform[Apply Resize 224x224 & Normalize]
    ApplyTransform --> RunInference[Execute Multi-Task Forward Pass on GPU/CPU]
    
    RunInference --> ExtractOutputs[Extract 6-Class Logits & 5x7x7 Grid Tensor]
    ExtractOutputs --> CompSoftmax[Compute Softmax Class Probabilities]
    ExtractOutputs --> DecodeGrid[Decode Grid Offsets to Normalized Coordinates]
    
    DecodeGrid --> FilterTau{Objectness >= tau?}
    FilterTau -- No --> DiscardCandidate[Discard Grid Cell]
    FilterTau -- Yes --> RetainCandidate[Retain Bounding Box Candidate]
    
    RetainCandidate --> RankSort[Sort Candidates Descending by Score]
    RankSort --> TruncateK[Truncate to Top-K Detections (K <= 3)]
    
    TruncateK --> PackJSON[Assemble JSON Response Payload]
    CompSoftmax --> PackJSON
    
    PackJSON --> Return200[Return HTTP 200 OK with Results]
    Return200 --> RenderUI[Render Disease Badge, Confidence Bars & Canvas Boxes]
    RenderUI --> End([Diagnosis Complete])

    style Start fill:#334155,color:#ffffff
    style End fill:#334155,color:#ffffff
    style ValClient fill:#1e293b,stroke:#f59e0b,color:#ffffff
    style ValServer fill:#1e293b,stroke:#f59e0b,color:#ffffff
    style FilterTau fill:#1e293b,stroke:#f59e0b,color:#ffffff
    style RunInference fill:#0f172a,stroke:#10b981,stroke-width:2px,color:#ffffff
    style RenderUI fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#ffffff
```
