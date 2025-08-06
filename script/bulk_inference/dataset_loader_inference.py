#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

class DatasetLoaderHook:
    @staticmethod
    def load_sampled_dataset(dataset_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Try to load a sampled dataset if it exists.
        Returns None if no sampled dataset is found.
        """
        cache_dir = Path("dataset_cache")
        sampled_file = cache_dir / f"{dataset_name}_sampled.json"
        
        if sampled_file.exists():
            try:
                with open(sampled_file, 'r') as f:
                    data = json.load(f)
                    return data.get('samples')
            except Exception as e:
                logger.error(f"Error loading sampled dataset {dataset_name}: {e}")
                return None
        return None

    @staticmethod
    def get_cache_path(dataset_name: str) -> Path:
        """Get the cache file path for a dataset"""
        return Path("dataset_cache") / f"{dataset_name}_sampled.json"