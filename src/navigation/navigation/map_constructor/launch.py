from .map_constructor import MapConstructor

CSV_PATH = ''
RESOLUTION = 0.05
OUTPUT_DIR = ''

# call init_map() before using get functions
mc = None


def init_map():
    """Create the MapConstructor, build the grid, and save outputs."""
    global mc
    mc = MapConstructor(
        csv_path=CSV_PATH,
        resolution=RESOLUTION,
    )
    mc.build()
    mc.save(OUTPUT_DIR)


def get_occupancy_grid():
    """Return a ROS OccupancyGrid message for the constructed map."""
    return mc.to_occupancy_grid()


def get_map_image():
    """Return a PIL Image of the constructed map."""
    return mc.to_image()
