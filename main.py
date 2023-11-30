# Constants and Configurations
from config import (
    INITIAL_CANVAS_WIDTH, INITIAL_CANVAS_HEIGHT, BASE_ALPHA_MIN, BASE_ALPHA_MAX
)

# Logger Setup
from logger import setup_logging, log_header, log_message

# Image Arrangement Algorithm
from image_arrangement import (
    visualize_solution,
    solve_optimal_arrangement, find_optimal_canvas,
    optimized_initial_arrangement
)


def main():
    setup_logging()
    log_header("NEW RUN")

    # Sample data for demonstration
    pictures = [(3133, 2126), (2532, 721), (2983, 2156), (961, 640), (980, 1132), (3149, 2102)]
    canvas_width, canvas_height = INITIAL_CANVAS_WIDTH, INITIAL_CANVAS_HEIGHT

    log_message("Using initial picture dimensions:", f"{pictures}")
    log_message("Initial canvas size:", f"{canvas_width}x{canvas_height}")

    pictures = optimized_initial_arrangement(pictures)

    # objective_types = ["unused_space", "max_dimension", "perimeter", "diff_width_height"]

    while pictures:
        log_message("Optimized picture arrangement:", f"{pictures}")

        optimal_canvas_width, optimal_canvas_height, _ = find_optimal_canvas(pictures, canvas_width, canvas_height, objective_type="diff_width_height")

        if optimal_canvas_width is None or optimal_canvas_height is None:
            log_message(f"No solution found for {len(pictures)} pictures.", "")
            # Remove the largest picture (which is the first one after sorting in optimized_initial_arrangement)
            pictures = pictures[1:]
        else:
            break

    if not pictures:
        log_message("No optimal canvas dimensions found even for a single picture", "")
        return
    
    log_message("Optimal canvas dimensions:", f"{optimal_canvas_width}x{optimal_canvas_height}")

    solution = solve_optimal_arrangement(pictures, optimal_canvas_width, optimal_canvas_height, objective_type="unused_space", log_search_progress= False, allow_scaling=True)

    # If there is no solution, iteratively remove the largest image until a solution is found
    while solution is None and len(pictures) > 0:
        smallest_image = min(pictures, key=lambda pic: pic[0]*pic[1])
        pictures.remove(smallest_image)
        solution = solve_optimal_arrangement(pictures, optimal_canvas_width, optimal_canvas_height, objective_type="unused_space", base_alpha_range=(BASE_ALPHA_MIN, BASE_ALPHA_MAX), allow_scaling=True, log_search_progress=False)

    if solution:
        visualize_solution(pictures, solution, optimal_canvas_width, optimal_canvas_height)
    else:
        log_message("No solution found", "")


if __name__ == "__main__":
    main()
