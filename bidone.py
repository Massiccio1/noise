import numpy as np
import plotly.graph_objs as go

def plot_3d_noisemap(matrix):
    """
    Plot a 3D noisemap from a 2D matrix using Plotly.

    Args:
    - matrix: 2D numpy array representing the noise map.

    Returns:
    - fig: Plotly figure object.
    """

    # Get the dimensions of the matrix
    rows, cols = matrix.shape

    # Create grid coordinates for the x and y axes
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    x, y = np.meshgrid(x, y)

    # Create the 3D surface plot
    fig = go.Figure(data=[go.Surface(z=matrix, x=x, y=y)])

    # Update the layout of the plot
    fig.update_layout(
        title='3D Noisemap',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Noise',
            camera_eye=dict(x=1.87, y=0.88, z=-0.64)
        )
    )

    return fig

# Example usage:
# Generate a random 2D noise map
matrix = np.random.rand(50, 50)

# Plot the 3D noisemap
fig = plot_3d_noisemap(matrix)

# Display the plot
fig.show()
