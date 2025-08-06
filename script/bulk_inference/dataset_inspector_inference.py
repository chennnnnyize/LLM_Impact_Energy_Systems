import promptbench as pb
import json

def inspect_dataset():
    # Load the dataset
    dataset = pb.DatasetLoader.load_dataset('bool_logic')
    
    # Print the first few examples
    print("\nFirst 3 examples from dataset:")
    for i, example in enumerate(dataset[:3]):
        print(f"\nExample {i + 1}:")
        print(json.dumps(example, indent=2))
    
    # Get schema information
    if len(dataset) > 0:
        print("\nKeys present in first example:")
        first_example = dataset[0]
        for key in first_example.keys():
            print(f"- {key}: {type(first_example[key]).__name__}")
    
    # Basic dataset statistics
    print(f"\nTotal number of examples: {len(dataset)}")
    
    # Check for consistency
    keys_per_example = [set(example.keys()) for example in dataset]
    all_keys = set().union(*keys_per_example)
    print(f"\nAll unique keys found across dataset: {all_keys}")
    
    # Check for missing keys
    if len(keys_per_example) > 0:
        consistent = all(keys == keys_per_example[0] for keys in keys_per_example)
        print(f"\nDataset has consistent keys across all examples: {consistent}")
        if not consistent:
            print("\nKey distribution:")
            key_counts = {}
            for keys in keys_per_example:
                key_tuple = tuple(sorted(keys))
                key_counts[key_tuple] = key_counts.get(key_tuple, 0) + 1
            for keys, count in key_counts.items():
                print(f"- Keys {keys}: {count} examples")

if __name__ == "__main__":
    inspect_dataset()
