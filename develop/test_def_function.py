from typing import Callable, List, Tuple, Optional, Any

# Define the function to load the initial mask
def load_initial_mask(initial_mask: list) -> Tuple[List[int], Optional[int]]:
    """Load the initial mask with some predefined values."""
    return initial_mask, None  # Return the initial mask and None for labels

# Define other example functions for the pipeline
def add(mask: list, value: int) -> Tuple[List[int], int]:
    mask.append(value)
    sum_value = sum(mask)
    return mask, sum_value

def subtract(mask: list, value: int) -> Tuple[List[int], int]:
    mask.append(value)
    sub_value = sum(mask) - value
    return mask, sub_value

def multiply(mask: list, value: int) -> Tuple[List[int], int]:
    mask.append(value)
    mul_value = sum(mask) * value
    return mask, mul_value

def more_args(mask: list, value: int, extra_value: Optional[int] = None) -> Tuple[List[int], int]:
    mask.append(value)
    if extra_value is not None:
        mask.append(extra_value)
    more_value = sum(mask) + (extra_value if extra_value is not None else 0)
    return mask, more_value

# Run pipeline function returning final results
def run_pipeline(steps: List[Tuple[str, Callable[..., Any], dict]], initial_mask: list) -> Tuple[List[int], Optional[int]]:
    # Call the first function to load the initial mask directly
    result = load_initial_mask(initial_mask)

    # Iterate through all steps
    for name, func, kwargs in steps:
        # Include the current mask in kwargs
        kwargs['mask'] = result[0]
        result = func(**kwargs)  # Call the function with unpacked kwargs

    # Return the final mask and the last computed value (if any)
    return result

# Initialize the mask with your chosen values
initial_mask = [1, 2, 3]

# Create a list of processing steps with named arguments
pipeline_steps: List[Tuple[str, Callable[..., Any], dict]] = [
    ("add", add, {'value': 5}),        
    ("subtract", subtract, {'value': 3}),          
    ("multiply", multiply, {'value': 4}),          
    ("more_args", more_args, {'value': 10, 'extra_value': 2}),  
]

# Run the defined pipeline and get the final results
final_results = run_pipeline(pipeline_steps, initial_mask)

# Output the final results
print(f"Final Mask: {final_results[0]}, Final Computed Value: {final_results[1] if final_results[1] is not None else 'N/A'}")
