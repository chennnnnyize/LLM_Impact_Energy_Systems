#!/usr/bin/env python3
import requests
import time
import promptbench as pb
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import logging
from pathlib import Path
import json
import sys
from dataset_processor import DatasetProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bulk_inference.log')
    ]
)
logger = logging.getLogger(__name__)

class BulkInferenceRunner:
    # Available math subsets based on inspection
    MATH_SUBSETS = [
        'arithmetic__add_or_sub',
        'arithmetic__div',
        'arithmetic__mul',
        'algebra__linear_1d',
        'algebra__linear_2d',
        'arithmetic__add_or_sub_in_base',
        'arithmetic__add_sub_multiple',
        'arithmetic__mixed',
        'arithmetic__mul_div_multiple'
    ]

    # Available language pairs for translation datasets
    TRANSLATION_PAIRS = {
        'un_multi': ['en-fr', 'en-es', 'en-ar', 'en-ru', 'en-zh'],
        'iwslt2017': ['en-de', 'en-fr', 'de-en', 'fr-en']
    }

    def __init__(
        self,
        sequential_url: str = "http://localhost:8000/run-sequential",
        parallel_url: str = "http://localhost:8000/run-parallel",
        batch_size: int = 32,
        output_dir: str = "results",
        timeout: int = 600,
        save_interval: int = 100,
        sampling_params: Optional[Dict[str, Any]] = None,
        use_sampled: bool = True
    ):
        self.sequential_url = sequential_url
        self.parallel_url = parallel_url
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.timeout = timeout
        self.save_interval = save_interval
        self.use_sampled = use_sampled
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure request parameters with defaults
        self.sampling_params = sampling_params or {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 4096
        }
        
        # Dataset-specific parameters
        self.processor = DatasetProcessor()

    def _get_math_subset(self, subset: Optional[str]) -> str:
        if not subset:
            return self.MATH_SUBSETS[0]
        if subset not in self.MATH_SUBSETS:
            available = '\n'.join(self.MATH_SUBSETS)
            raise ValueError(
                f"Invalid math subset: {subset}\n"
                f"Available subsets:\n{available}"
            )
        return subset

    def _get_translation_pair(self, dataset_name: str, pair: Optional[str]) -> str:
        if dataset_name not in self.TRANSLATION_PAIRS:
            raise ValueError(f"Unknown translation dataset: {dataset_name}")
        
        available_pairs = self.TRANSLATION_PAIRS[dataset_name]
        if not pair:
            return available_pairs[0]
        if pair not in available_pairs:
            pairs_str = '\n'.join(available_pairs)
            raise ValueError(
                f"Invalid language pair for {dataset_name}: {pair}\n"
                f"Available pairs:\n{pairs_str}"
            )
        return pair

    def load_dataset(self, dataset_name: str, subset: Optional[str] = None) -> List[Dict]:
        try:
            if self.use_sampled:
                from dataset_loader import DatasetLoaderHook
                sampled_data = DatasetLoaderHook.load_sampled_dataset(dataset_name)
                if sampled_data is not None:
                    logger.info(f"Using sampled dataset for {dataset_name}")
                    return sampled_data
                logger.warning(f"No sampled dataset found for {dataset_name}, falling back to full dataset")
        
            logger.info(f"Loading full dataset for {dataset_name}")
            return pb.DatasetLoader.load_dataset(dataset_name, subset)
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise

    def _process_response_with_metrics(
        self,
        dataset_name: str,
        item: Dict[str, Any],
        response: str,
        formatted_prompt: str
    ) -> Dict[str, Any]:
        try:
            processed_output, is_valid = self.processor.process_response(
                dataset_name, response, item
            )
            
            result = {
                "input": item,
                "formatted_prompt": formatted_prompt,
                "raw_output": response,
                "processed_output": processed_output,
                "is_valid_format": is_valid,
                "status": "success"
            }
            
            if dataset_name == "sst2" and is_valid:
                result["correct"] = processed_output == item.get("label")
            elif dataset_name in ["bool_logic", "valid_parentheses"] and is_valid:
                result["correct"] = str(processed_output).lower() == str(item.get("label")).lower()
            elif dataset_name == "mmlu" and is_valid:
                result["correct"] = processed_output == item.get("answer")
            elif dataset_name.startswith("math") and is_valid:
                expected = item["answer"].replace("b'", "").replace("\\n'", "")
                try:
                    result["correct"] = abs(float(processed_output) - float(expected)) < 1e-6
                except (ValueError, TypeError):
                    result["correct"] = False
            
            return result
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            return {
                "input": item,
                "formatted_prompt": formatted_prompt,
                "error": str(e),
                "status": "error"
            }

    def process_batch_sequential(
        self, 
        batch: List[Dict[str, Any]], 
        dataset_name: str
    ) -> List[Dict[str, Any]]:
        """Process a batch of prompts sequentially."""
        results = []
        for item in tqdm(batch, desc="Processing sequentially"):
            try:
                validation_error = self.processor.validate_dataset_example(dataset_name, item)
                if validation_error:
                    logger.warning(f"Skipping invalid example: {validation_error}")
                    continue

                formatted_prompt = self.processor.format_prompt(dataset_name, item)
                logger.debug(f"Sending prompt: {formatted_prompt}")
                
                request_data = {
                    "prompts": [formatted_prompt],
                    "sampling_params": self.sampling_params
                }
                
                response = requests.post(
                    self.sequential_url,
                    json=request_data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    model_response = response.json()["responses"][0]
                    logger.debug(f"Received response: {model_response}")
                    
                    result = self._process_response_with_metrics(
                        dataset_name, item, model_response, formatted_prompt
                    )
                    results.append(result)
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    logger.error(f"Request failed: {error_msg}")
                    results.append({
                        "input": item,
                        "formatted_prompt": formatted_prompt,
                        "error": error_msg,
                        "status": "error"
                    })
            except Exception as e:
                logger.error(f"Error processing item: {str(e)}")
                results.append({
                    "input": item,
                    "error": str(e),
                    "status": "error"
                })
            time.sleep(1)  # Rate limiting
        return results

    def process_batch_parallel(
        self, 
        batch: List[Dict[str, Any]], 
        dataset_name: str,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """Process a batch of prompts in parallel with retry logic."""
        attempt = 0
        while attempt < max_retries:
            try:
                formatted_prompts = []
                valid_items = []
                
                for item in batch:
                    validation_error = self.processor.validate_dataset_example(dataset_name, item)
                    if validation_error:
                        logger.warning(f"Skipping invalid example: {validation_error}")
                        continue
                        
                    formatted_prompts.append(self.processor.format_prompt(dataset_name, item))
                    valid_items.append(item)
                
                if not formatted_prompts:
                    logger.warning("No valid prompts in batch")
                    return []
                
                logger.debug(f"Processing batch of {len(formatted_prompts)} prompts")
                
                request_data = {
                    "prompts": formatted_prompts,
                    "sampling_params": self.sampling_params
                }
                
                response = requests.post(
                    self.parallel_url,
                    json=request_data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    model_responses = response.json()["responses"]
                    logger.debug(f"Received {len(model_responses)} responses")
                    
                    return [
                        self._process_response_with_metrics(
                            dataset_name, item, resp, prompt
                        )
                        for item, prompt, resp in zip(valid_items, formatted_prompts, model_responses)
                    ]
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    logger.error(f"Batch request failed: {error_msg}")
                    return [
                        {
                            "input": item,
                            "formatted_prompt": prompt,
                            "error": error_msg,
                            "status": "error"
                        }
                        for item, prompt in zip(valid_items, formatted_prompts)
                    ]

            except requests.Timeout:
                attempt += 1
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"Timeout occurred. Attempt {attempt}/{max_retries}. "
                                 f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries ({max_retries}) exceeded for batch")
                    raise
                    
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                return [
                    {
                        "input": item,
                        "error": str(e),
                        "status": "error"
                    }
                    for item in valid_items
                ]
        
        return [
            {
                "input": item,
                "error": "Max retries exceeded",
                "status": "error"
            }
            for item in valid_items
        ]

    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        metrics = {
            "success_rate": 0.0,
            "valid_format_rate": 0.0,
            "accuracy": 0.0
        }
        
        total = len(results)
        if total == 0:
            return metrics
        
        successful = sum(1 for r in results if r["status"] == "success")
        valid_format = sum(1 for r in results if r["status"] == "success" and r.get("is_valid_format", False))
        correct = sum(1 for r in results if r.get("correct", False))
        
        metrics["success_rate"] = (successful / total) * 100
        metrics["valid_format_rate"] = (valid_format / total) * 100
        if valid_format > 0:
            metrics["accuracy"] = (correct / valid_format) * 100
            
        return metrics

    def run_inference(
        self,
        dataset_name: str,
        subset: str = None,
        mode: str = "sequential",
        save_interval: int = 100
    ) -> None:
        try:
            dataset = self.load_dataset(dataset_name, subset)
            results = []
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = self.output_dir / f"{dataset_name}_{mode}_{timestamp}.jsonl"
            
            logger.info(f"Starting {mode} inference on {dataset_name}")
            logger.info(f"Results will be saved to: {output_file}")
            
            total_examples = len(dataset)
            progress_bar = tqdm(total=total_examples, desc=f"Processing {dataset_name}")
            
            for i in range(0, total_examples, self.batch_size):
                batch = dataset[i:i + self.batch_size]
                batch_size = len(batch)
                
                try:
                    if mode == "sequential":
                        batch_results = self.process_batch_sequential(batch, dataset_name)
                    else:
                        batch_results = self.process_batch_parallel(batch, dataset_name)
                    
                    results.extend(batch_results)
                    progress_bar.update(batch_size)
                    
                    if len(results) % save_interval == 0:
                        self._save_results(results, output_file)
                        current_success = sum(1 for r in results if r["status"] == "success")
                        current_valid = sum(1 for r in results if r["status"] == "success" and r.get("is_valid_format", False))
                        logger.info(
                            f"Progress: {len(results)}/{total_examples} examples processed. "
                            f"Success rate: {current_success/len(results)*100:.2f}%, "
                            f"Valid format rate: {current_valid/len(results)*100:.2f}%"
                        )
                
                except Exception as e:
                    logger.error(f"Error processing batch starting at index {i}: {str(e)}")
                    continue
            
            progress_bar.close()
            
            self._save_results(results, output_file)
            logger.info(f"Completed processing {len(results)} examples. Results saved to {output_file}")
            
            self._log_statistics(results)
            
        except Exception as e:
            logger.error(f"Fatal error during inference run: {str(e)}")
            raise

    def _save_results(self, results: List[Dict[str, Any]], output_file: Path) -> None:
        try:
            with open(output_file, 'w') as f:
                for result in results:
                    json_str = json.dumps(result)
                    f.write(json_str + '\n')
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {str(e)}")
            raise

    def _log_statistics(self, results: List[Dict[str, Any]]) -> None:
        metrics = self._calculate_metrics(results)
        
        logger.info("\n=== Inference Statistics ===")
        logger.info(f"Total processed: {len(results)}")
        logger.info(f"Success rate: {metrics['success_rate']:.2f}%")
        logger.info(f"Valid format rate: {metrics['valid_format_rate']:.2f}%")
        if metrics['accuracy'] > 0:
            logger.info(f"Accuracy: {metrics['accuracy']:.2f}%")
        
        failed = sum(1 for r in results if r["status"] == "error")
        if failed > 0:
            error_categories = {}
            for result in results:
                if result["status"] == "error":
                    error_msg = result.get("error", "Unknown error")
                    error_type = error_msg.split(':')[0]
                    error_categories[error_type] = error_categories.get(error_type, 0) + 1
                    
            logger.info("\nError Categories:")
            for error_type, count in error_categories.items():
                logger.info(f"{error_type}: {count} ({count/failed*100:.2f}% of failures)")

def main():
    parser = argparse.ArgumentParser(
        description="Run bulk inference on datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add the use-sampled-data argument
    parser.add_argument(
        "--use-sampled-data",
        action="store_true",
        help="Use sampled dataset instead of full dataset"
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help="Dataset name to process"
    )
    
    # Optional arguments
    parser.add_argument(
        "--subset", 
        type=str, 
        help="Dataset subset (if applicable)"
    )
    parser.add_argument(
        "--mode", 
        choices=["sequential", "parallel"], 
        default="sequential",
        help="Processing mode: sequential or parallel"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=300,
        help="Timeout for each request in seconds"
    )
    parser.add_argument(
        "--save-interval", 
        type=int, 
        default=100,
        help="Save intermediate results every N examples"
    )
    parser.add_argument(
        "--sequential-url",
        type=str,
        default="http://localhost:8000/run-sequential",
        help="URL for sequential inference endpoint"
    )
    parser.add_argument(
        "--parallel-url",
        type=str,
        default="http://localhost:8000/run-parallel",
        help="URL for parallel inference endpoint"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    try:
        # Print available datasets if running promptbench
        try:
            logger.info("Available datasets:")
            logger.info(pb.SUPPORTED_DATASETS)
        except Exception as e:
            logger.warning(f"Could not list available datasets: {e}")

        # Initialize runner once with all arguments
        runner = BulkInferenceRunner(
            sequential_url=args.sequential_url,
            parallel_url=args.parallel_url,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            timeout=args.timeout,
            use_sampled=args.use_sampled_data
        )

        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
    
        # Log configuration
        logger.info("\nRunning with configuration:")
        logger.info(f"Dataset: {args.dataset}")
        if args.subset:
            logger.info(f"Subset: {args.subset}")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Save interval: {args.save_interval}")
        logger.info(f"Timeout: {args.timeout}s")
        if args.mode == "sequential":
            logger.info(f"Sequential URL: {args.sequential_url}")
        else:
            logger.info(f"Parallel URL: {args.parallel_url}")
    
        # Run inference
        start_time = time.time()
        runner.run_inference(
            dataset_name=args.dataset,
            subset=args.subset,
            mode=args.mode,
            save_interval=args.save_interval
        )
        end_time = time.time()
    
        # Log completion time
        duration = end_time - start_time
        logger.info(f"\nProcessing completed in {duration:.2f} seconds")
    
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        if args.debug:
            logger.exception("Detailed traceback:")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        sys.exit(1)