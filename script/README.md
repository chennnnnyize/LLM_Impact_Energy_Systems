# Coming soon ...

graph LR
    A[script] --> B[README.md]
    A --> C[bulk_inference]
    A --> D[finetune]
    A --> E[power_monitor]
    A --> F[rag]
    A --> G[train]

    %% bulk_inference contents
    subgraph bulk_inference
        C1[bulk_inference.py]
        C2[dataset_inspector_inference.py]
        C3[dataset_loader_inference.py]
        C4[dataset_processor_inference.py]
        C5[dataset_sampler_inference.py]
        C6[model_server_inference.py]
        C7[requirements_inference.txt]
        C8[inference_data/]
        C9[inference_power_consumption/]
    end

    %% power_monitor contents
    subgraph power_monitor
        E1[amd_gpu_csv.py]
        E2[intel_cpu_csv.py]
        E3[nvidia_gpu_csv.py]
    end

    %% train contents
    subgraph train
        G1[local_training_script_llm.c.txt]
        G2[train_data/]
        G3[train_power_consumption/]
    end
