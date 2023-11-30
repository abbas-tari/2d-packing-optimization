import sys
import os
from contextlib import contextmanager
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.style as style
import numpy as np
from logger import log_message
from utils import adjust_alpha_range
from config import (
    INITIAL_CANVAS_WIDTH, INITIAL_CANVAS_HEIGHT, BASE_ALPHA_MIN, BASE_ALPHA_MAX
)

@contextmanager
def redirect_stdout_to_file(filename):
    """
    Context manager for redirecting stdout to a file.
    """
    original_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f
        yield
        sys.stdout = original_stdout

def visualize_solution(pictures: list, solution: list, canvas_width: int, canvas_height: int):
    """Visualize the solution using matplotlib with a fancy design."""
    # Use a modern style for the plot
    style.use('ggplot')  # or 'seaborn', 'fivethirtyeight', etc.

    fig, ax = plt.subplots()
    ax.set_xlim([0, canvas_width])
    ax.set_ylim([0, canvas_height])
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Origin is top-left corner

    # Use a colormap for colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(pictures)))

    for i, (x_val, y_val, scale) in enumerate(solution):
        rect = patches.Rectangle((x_val, y_val),
                                pictures[i][0] * scale,
                                pictures[i][1] * scale,
                                edgecolor='black',  # Add contrast
                                facecolor=colors[i])
        ax.add_patch(rect)
        ax.text(x_val + pictures[i][0] * scale / 2,
                y_val + pictures[i][1] * scale / 2,
                f"Image {i}",
                ha='center', va='center', fontsize=10, color='white',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    # Add a title and improve layout
    plt.title("Solution Visualization", fontsize=14)
    plt.tight_layout()

    # Save the plot with high resolution
    plt.savefig('solution_visualization.png', dpi=300)

    plt.show()


def add_non_overlapping_constraints(model, x, y, widths, heights):
    """
    Add non-overlapping constraints to the model.
    
    Args:
        model: The constraint programming model instance.
        x, y: Lists of x and y coordinates of images.
        widths, heights: Lists of widths and heights of images.
    """
    n = len(x)
    for i in range(n):
        for j in range(n):
            if i != j:
                # If image i is to the left of image j
                left = model.NewBoolVar(f'left_{i}_{j}')
                model.Add(x[i] + widths[i] <= x[j]).OnlyEnforceIf(left)  

                # If image i is to the right of image j
                right = model.NewBoolVar(f'right_{i}_{j}')
                model.Add(x[j] + widths[j] <= x[i]).OnlyEnforceIf(right) 

                # If image i is above image j
                above = model.NewBoolVar(f'above_{i}_{j}')
                model.Add(y[i] + heights[i] <= y[j]).OnlyEnforceIf(above)  

                # If image i is below image j
                below = model.NewBoolVar(f'below_{i}_{j}')
                model.Add(y[j] + heights[j] <= y[i]).OnlyEnforceIf(below)  

                # At least one of the conditions must be true
                model.AddBoolOr([left, right, above, below])


def solve_optimal_arrangement(pictures, canvas_width, canvas_height, objective_type="unused_space", base_alpha_range=(BASE_ALPHA_MIN, BASE_ALPHA_MAX), allow_scaling=True, num_search_workers=8, log_search_progress=False):
    """
    Solve the optimization problem with the option to scale the images.
    
    Args:
        pictures: List of image dimensions.
        canvas_width, canvas_height: Dimensions of the canvas.
        objective_type: Objective function type for the optimization.
        base_alpha_range: Base alpha range.
        allow_scaling: Boolean indicating if image scaling is allowed.
        num_search_workers: Number of search workers for the solver.
        log_search_progress: Boolean indicating if solver progress should be logged.

    Returns:
        List of solutions containing image positions and scaling factors.
    """
    model = cp_model.CpModel()
    n = len(pictures)
    bounding_box_area = canvas_width * canvas_height
    
    x = [model.NewIntVar(0, canvas_width, f'x_{i}') for i in range(n)]
    y = [model.NewIntVar(0, canvas_height, f'y_{i}') for i in range(n)]
    
    if allow_scaling:
        # Adjust alpha range for each picture based on its size relative to the bounding box
        alpha_ranges = [adjust_alpha_range(pictures[i], bounding_box_area, base_alpha_range) for i in range(n)]
        alpha = [model.NewIntVar(alpha_range[0], alpha_range[1], f'alpha_{i}') for i, alpha_range in enumerate(alpha_ranges)]
        
        scaled_widths = [model.NewIntVar(0, canvas_width, f'scaled_width_{i}') for i in range(n)]
        scaled_heights = [model.NewIntVar(0, canvas_height, f'scaled_height_{i}') for i in range(n)]
        
        # Constraints to represent the scaled dimensions
        for i in range(n):
            model.AddDivisionEquality(scaled_widths[i], pictures[i][0] * alpha[i], 100)
            model.AddDivisionEquality(scaled_heights[i], pictures[i][1] * alpha[i], 100)
        
        # Constraints to make sure images are within canvas bounds
        for i in range(n):
            model.Add(x[i] + scaled_widths[i] <= canvas_width)
            model.Add(y[i] + scaled_heights[i] <= canvas_height)
        
        add_non_overlapping_constraints(model, x, y, scaled_widths, scaled_heights)
    
    else:
        # When scaling is not allowed
        widths = [pictures[i][0] for i in range(n)]
        heights = [pictures[i][1] for i in range(n)]

        # Constraints to make sure images are within canvas bounds
        for i in range(n):
            model.Add(x[i] + widths[i] <= canvas_width)
            model.Add(y[i] + heights[i] <= canvas_height)
            
        add_non_overlapping_constraints(model, x, y, widths, heights)

    # Objective to make bounding box as square as possible
    max_x = model.NewIntVar(0, canvas_width, 'max_x')
    max_y = model.NewIntVar(0, canvas_height, 'max_y')
    if allow_scaling:
        model.AddMaxEquality(max_x, [x[i] + scaled_widths[i] for i in range(n)]) 
        model.AddMaxEquality(max_y, [y[i] + scaled_heights[i] for i in range(n)]) 
    else:
        model.AddMaxEquality(max_x, [x[i] + widths[i] for i in range(n)])  
        model.AddMaxEquality(max_y, [y[i] + heights[i] for i in range(n)])
    
    # Selecting the objective function
    if objective_type == "unused_space":
        if allow_scaling:
            product_vars = [model.NewIntVar(0, canvas_width * canvas_height, f'product_{i}') for i in range(n)]
            for i in range(n):
                model.AddMultiplicationEquality(product_vars[i], [scaled_widths[i], scaled_heights[i]])
            total_picture_area = sum(product_vars)
        else:
            total_picture_area = sum([widths[i] * heights[i] for i in range(n)])
        model.Minimize(1*(bounding_box_area - total_picture_area) + 0*(max_x + max_y))
    elif objective_type == "max_dimension":
        bounding_box_max_dim = model.NewIntVar(0, canvas_width, 'bounding_box_max_dim')
        model.AddMaxEquality(bounding_box_max_dim, [max_x, max_y])
        model.Minimize(bounding_box_max_dim)
    elif objective_type == "perimeter":
        model.Minimize(max_x + max_y)
    elif objective_type == "diff_width_height":
        diff_dims = model.NewIntVar(0, canvas_width, 'diff_dims')
        model.AddAbsEquality(diff_dims, max_x - max_y)
        model.Minimize(diff_dims)
    else:
        raise ValueError("Invalid objective_type.")
    
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = num_search_workers
    solver.parameters.log_search_progress = log_search_progress
    

    if log_search_progress:
        log_filename = 'solve_problem_log.txt' if allow_scaling else 'solve_problem_full_size_log.txt'
        log_message("Solver progress is being logged to", log_filename)

        # Redirect stdout to log file
        original_stdout_fd = os.dup(1)
        log_file_fd = os.open(log_filename, os.O_WRONLY | os.O_CREAT)
        os.dup2(log_file_fd, 1)

        status = solver.Solve(model)

        # Restore stdout
        os.dup2(original_stdout_fd, 1)
        os.close(log_file_fd)
    else:
        status = solver.Solve(model)



    if status == cp_model.OPTIMAL:
        solution = []
        for i in range(n):
            if allow_scaling:
                solution.append((solver.Value(x[i]), solver.Value(y[i]), solver.Value(alpha[i])/100.0))  
                log_message(f"Alpha/Scaling factor for Image {i}", f"{solver.Value(alpha[i])}%")
            else:
                solution.append((solver.Value(x[i]), solver.Value(y[i]), 1.0))
        return solution
    else:
        return None

def find_optimal_canvas(pictures, initial_canvas_width=INITIAL_CANVAS_WIDTH, initial_canvas_height=INITIAL_CANVAS_HEIGHT, objective_type="perimeter"):
    """
    Find the optimal canvas dimensions using a binary search approach.

    Args:
        pictures (List[Tuple[int, int]]): List of image dimensions.
        initial_canvas_width (int): Initial width of the canvas.
        initial_canvas_height (int): Initial height of the canvas.
        objective_type (str): Objective function type for the optimization.

    Returns:
        Tuple[int, int, List[Tuple[int, int, float]]]: Optimal canvas dimensions and the solution.
    """
    canvas_width_low, canvas_width_high = min(pic[0] for pic in pictures), initial_canvas_width
    canvas_height_low, canvas_height_high = min(pic[1] for pic in pictures), initial_canvas_height

    best_canvas_width, best_canvas_height = None, None
    best_solution = None  # Initialize variable to store the best solution
    min_diff = float('inf')

    while canvas_width_low <= canvas_width_high and canvas_height_low <= canvas_height_high:
        canvas_width_mid = (canvas_width_low + canvas_width_high) // 2
        canvas_height_mid = (canvas_height_low + canvas_height_high) // 2

        solution = solve_optimal_arrangement(pictures, canvas_width_mid, canvas_height_mid, objective_type=objective_type, log_search_progress= False, allow_scaling=False)
        log_message("Trying canvas dimensions:", f"{canvas_width_mid}x{canvas_height_mid}. Solution found: {'Yes' if solution else 'No'}")
        current_diff = abs(canvas_width_mid - canvas_height_mid)

        if solution and len(solution) == len(pictures):
            if current_diff <= min_diff:
                min_diff = current_diff
                best_canvas_width, best_canvas_height = canvas_width_mid, canvas_height_mid
                best_solution = solution  # Update the best solution

            if canvas_width_mid > canvas_height_mid:
                canvas_width_high = canvas_width_mid - 1
            elif canvas_height_mid > canvas_width_mid:
                canvas_height_high = canvas_height_mid - 1
            else:
                canvas_width_high = canvas_width_mid - 1
                canvas_height_high = canvas_height_mid - 1
        else:
            if canvas_width_mid > canvas_height_mid:
                canvas_width_low = canvas_width_mid + 1
            elif canvas_height_mid > canvas_width_mid:
                canvas_height_low = canvas_height_mid + 1
            else:
                canvas_width_low = canvas_width_mid + 1
                canvas_height_low = canvas_height_mid + 1

    return best_canvas_width, best_canvas_height, best_solution  # Return the stored best solution

def optimized_initial_arrangement(pictures):
    """
    Optimize the initial arrangement of pictures.

    Args:
        pictures (List[Tuple[int, int]]): List of image dimensions.

    Returns:
        List[Tuple[int, int]]: Optimized arrangement of pictures.
    """
    # Sort the pictures in decreasing order based on their area (width x height)
    sorted_pictures = sorted(pictures, key=lambda pic: pic[0]*pic[1], reverse=True)
    return sorted_pictures
