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

    %% finetune contents
    subgraph finetune
        D1[command_2024_11_04.txt]
        D2[finetune_command.txt]
        D3[finetune_custom_dataset.py]
        D4[finetune_custom_dataset_2024_11_04.py]
        D5[finetune_datasets.py]
        D6[finetune_finetuning.py]
        D7[finetune_script_llama_recipes.txt]
        D8[finetune_training.py]
        D9[finetune__init__.py]
        D10[finetune_data/]
        D11[finetune_power_consumption/]
    end

    %% rag contents
    subgraph rag
        F1[rag_automation.py]
        F2[rag_bulk_query_test.py]
        F3[rag_check_setup.py]
        F4[rag_document_processor.py]
        F5[rag_model_server.py]
        F6[rag_process_sample_file.py]
        F7[rag_query_client.py]
        F8[rag_requirements.txt]
        F9[rag_run_bulk_test.py]
        F10[rag_run_parallel_test.py]
        F11[rag_service.py]
        F12[rag_setup_check.py]
        F13[rag_vllm_monitor.py]
    end
