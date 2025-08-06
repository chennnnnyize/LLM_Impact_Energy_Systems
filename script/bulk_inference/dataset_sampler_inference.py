#!/usr/bin/env python3
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import promptbench as pb

logger = logging.getLogger(__name__)

class DatasetPreprocessor:
    """Handles dataset sampling and preparation"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.samples_dir = self.base_dir / "sampled_datasets"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def convert_to_list(dataset) -> List[Dict]:
        """Convert dataset to list format regardless of input type."""
        if isinstance(dataset, list):
            return dataset
        try:
            # Try converting to list directly
            return list(dataset)
        except TypeError:
            # If direct conversion fails, try accessing as dictionary or dataset object
            if hasattr(dataset, 'to_dict'):
                return list(dataset.to_dict())
            if hasattr(dataset, 'to_list'):
                return dataset.to_list()
            if hasattr(dataset, '__getitem__') and hasattr(dataset, '__len__'):
                return [dataset[i] for i in range(len(dataset))]
            raise TypeError(f"Unable to convert dataset of type {type(dataset)} to list")

    def prepare_individual_samples(self, 
                                 dataset_configs: List[Dict], 
                                 samples_per_dataset: int = 100) -> Dict[str, str]:
        """
        Sample each dataset individually and save them.
        Returns dict mapping dataset names to their sampled data paths.
        """
        sampled_paths = {}
        
        for config in dataset_configs:
            dataset_name = config["name"]
            subset = config.get("subset")
            
            try:
                # Load and sample dataset
                logger.info(f"Loading dataset {dataset_name}...")
                dataset = pb.DatasetLoader.load_dataset(dataset_name, subset)
                logger.info(f"Converting {dataset_name} to list format...")
                dataset_list = self.convert_to_list(dataset)
                logger.info(f"Dataset {dataset_name} has {len(dataset_list)} examples")
                sample_size = min(samples_per_dataset, len(dataset_list))
                logger.info(f"Sampling {sample_size} examples from {dataset_name}...")
                sampled_data = random.sample(dataset_list, sample_size)
                
                # Add dataset identifier to each example
                for item in sampled_data:
                    item['dataset_source'] = dataset_name
                
                # Save sampled dataset
                output_path = self.samples_dir / f"{dataset_name}_individual_samples.json"
                with open(output_path, 'w') as f:
                    json.dump({
                        "dataset": dataset_name,
                        "subset": subset,
                        "original_size": len(dataset),
                        "sampled_size": len(sampled_data),
                        "samples": sampled_data
                    }, f, indent=2)
                
                sampled_paths[dataset_name] = str(output_path)
                logger.info(f"Saved {len(sampled_data)} samples from {dataset_name} to {output_path}")
                
            except Exception as e:
                logger.error(f"Error sampling dataset {dataset_name}: {str(e)}")
                continue
        
        return sampled_paths

    def prepare_mixed_dataset(self, 
                            dataset_configs: List[Dict], 
                            samples_per_dataset: int = 100) -> str:
        """
        Create a mixed dataset with samples from all datasets.
        Returns path to the mixed dataset file.
        """
        all_samples = []
        
        for config in dataset_configs:
            dataset_name = config["name"]
            subset = config.get("subset")
            
            try:
                # Load and sample dataset
                dataset = pb.DatasetLoader.load_dataset(dataset_name, subset)
                # Convert dataset to list if it's not already
                dataset_list = list(dataset) if not isinstance(dataset, list) else dataset
                sampled_data = random.sample(dataset_list, min(samples_per_dataset, len(dataset_list)))
                
                # Add dataset identifier to each example
                for item in sampled_data:
                    item['dataset_source'] = dataset_name
                
                all_samples.extend(sampled_data)
                logger.info(f"Added {len(sampled_data)} samples from {dataset_name} to mixed dataset")
                
            except Exception as e:
                logger.error(f"Error sampling dataset {dataset_name}: {str(e)}")
                continue
        
        # Shuffle all samples
        random.shuffle(all_samples)
        
        # Save mixed dataset
        output_path = self.samples_dir / "mixed_dataset_samples.json"
        with open(output_path, 'w') as f:
            json.dump({
                "total_samples": len(all_samples),
                "samples_per_dataset": samples_per_dataset,
                "samples": all_samples
            }, f, indent=2)
        
        logger.info(f"Saved mixed dataset with {len(all_samples)} total samples to {output_path}")
        return str(output_path)

class BatchGenerator:
    """Handles batch creation and random delays"""
    
    def __init__(self, min_delay: int = 1, max_delay: int = 5):
        self.min_delay = min_delay
        self.max_delay = max_delay
    
    def get_random_delay(self) -> float:
        """Generate random delay in seconds"""
        return random.uniform(self.min_delay, self.max_delay)
    
    def create_batches(self, data: List[Dict], batch_size: int) -> List[List[Dict]]:
        """Split data into batches"""
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

class DatasetManager:
    """Main class for managing dataset operations"""
    
    def __init__(self, base_dir: Path, dataset_configs: List[Dict]):
        self.base_dir = Path(base_dir)
        self.dataset_configs = dataset_configs
        self.preprocessor = DatasetPreprocessor(base_dir)
        self.batch_generator = BatchGenerator()
        
    def prepare_all_datasets(self, samples_per_dataset: int = 100) -> Dict[str, Any]:
        """
        Prepare both individual and mixed datasets.
        Returns paths to all prepared datasets.
        """
        logger.info("Starting dataset preparation...")
        
        # Prepare individual samples
        individual_paths = self.preprocessor.prepare_individual_samples(
            self.dataset_configs, 
            samples_per_dataset
        )
        
        # Prepare mixed dataset
        mixed_path = self.preprocessor.prepare_mixed_dataset(
            self.dataset_configs,
            samples_per_dataset
        )
        
        return {
            "individual_samples": individual_paths,
            "mixed_samples": mixed_path
        }

def load_samples(file_path: str) -> List[Dict]:
    """Utility function to load sampled data"""
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data["samples"]