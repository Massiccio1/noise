from opensimplex import OpenSimplex

def generate_simplex_noise(width, height, scale=1.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=0):
    """Generate a simplex noise matrix.

    Args:
        width (int): Width of the noise matrix.
        height (int): Height of the noise matrix.
        scale (float): Scaling factor for the noise.
        octaves (int): Number of octaves in the noise generation.
        persistence (float): Persistence value for the noise generation.
        lacunarity (float): Lacunarity value for the noise generation.
        seed (int): Seed for the noise generation.

    Returns:
        list: 2D simplex noise matrix.
    """
    noise = OpenSimplex(seed=seed)
    noise_matrix = []

    for y in range(height):
        row = []
        for x in range(width):
            nx = x / width - 0.5
            ny = y / height - 0.5

            value = 0.0
            amplitude = 1.0
            total_amplitude = 0.0

            for _ in range(octaves):
                value += noise.noise2(nx * scale, ny * scale) * amplitude
                total_amplitude += amplitude
                amplitude *= persistence
                scale *= lacunarity

            value /= total_amplitude
            row.append(value)

        noise_matrix.append(row)

    return noise_matrix

# Example usage:
width = 100
height = 100
scale = 0.1
octaves = 6
persistence = 0.5
lacunarity = 2.0
seed = 42

simplex_noise = generate_simplex_noise(width, height, scale, octaves, persistence, lacunarity, seed)
print(simplex_noise)