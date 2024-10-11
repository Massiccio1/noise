import noise
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import random
# from scipy.signal import convolve2d
import datetime
import os
from functools import wraps
import time
import opensimplex
from opensimplex import OpenSimplex
import math

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper



def generate_noise_map(width, height, scale, octaves, persistence, lacunarity, seed):
    """
    Generate a 2D noise map using Perlin noise algorithm.

    Parameters:
    - width: Width of the noise map.
    - height: Height of the noise map.
    - scale: Scale of the noise map.
    - octaves: Number of octaves in the Perlin noise algorithm.
    - persistence: Persistence parameter in the Perlin noise algorithm.
    - lacunarity: Lacunarity parameter in the Perlin noise algorithm.
    - seed: Seed value for the random number generator.

    Returns:
    - noise_map: A 2D numpy array representing the generated noise map.
    """
    noise_map = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            noise_map[y][x] = noise.pnoise3(x / scale,
                                             y / scale,
                                             seed,
                                             octaves=octaves,
                                             persistence=persistence,
                                             lacunarity=lacunarity,
                                             )
    return noise_map

    
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
                print(f"{nx * scale} {ny * scale}")
                value += noise.noise2(nx * scale, ny * scale) * amplitude
                total_amplitude += amplitude
                amplitude *= persistence
                scale *= lacunarity

            value /= total_amplitude
            row.append(value)

        noise_matrix.append(row)

    return noise_matrix



def save_noise_map_as_png(noise_map, filename):
    """
    Save the noise map as a PNG image.

    Parameters:
    - noise_map: The 2D numpy array representing the noise map.
    - filename: The filename to save the PNG image.
    """
    noise_map = (noise_map + 1) * 127.5  # Convert from [-1, 1] to [0, 255]
    noise_map = noise_map.astype(np.uint8)
    img = Image.fromarray(noise_map, 'L')
    
    if os.path.isfile(filename):
        os.remove(filename)
        print("removing...")
    
    img.save(filename)

def plot(noise_map):
    nrows, ncols = noise_map.shape
    marker_data = go.Mesh3d(
        x=np.arange(0,ncols, dtype=int), 
        y=np.arange(0,nrows, dtype=int), 
        z=noise_map, 
        opacity=0.8, 
    )
    fig=go.Figure(data=marker_data)
    fig.show()

def hemisphere(h,w):
    noise_map = np.zeros((h, w))
    center = (h/2,w/2)
    radius = np.sqrt((w/2*h/2))
    radius2 = (w/2*h/2)
    print(center)
    for y in range(h):
        for x in range(w):
            # noise_map[y][x]= np.sqrt(((y-center[0])**2 + (x-center[1])**2 )) / radius
            noise_map[y][x] = np.sqrt(radius2 - (y-(h/2))**2 - (x-(w/2))**2) 
    noise_map[np.isnan(noise_map)] = 0
    return noise_map

    
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
    x = np.linspace(0, cols, cols)
    y = np.linspace(0, rows, rows)
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
            camera_eye=dict(x=1.87, y=0.88, z=1.64)
        )
        
    )

    # fig = plot_3d_noisemap(matrix)

    # Display the plot
    fig.show()
    
# def low_pass_filter(matrix, kernel_size=3):
#     """
#     Apply a low-pass filter to a 2D matrix using convolution.

#     Args:
#     - matrix: 2D numpy array representing the input matrix.
#     - kernel_size: Size of the filter kernel (default is 3x3).

#     Returns:
#     - filtered_matrix: 2D numpy array representing the filtered matrix.
#     """

#     # Define a low-pass filter kernel
#     kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

#     # Apply convolution to the input matrix using the filter kernel
#     filtered_matrix = convolve2d(matrix, kernel, mode='same', boundary='wrap')

#     return filtered_matrix

 

def flood(matrix, perc, verbose = False):
    x,y = matrix.shape
    tot = x*y
    max = np.max(matrix)
    min = np.min(matrix)
    mean = np.mean(matrix)
    thr = (max-min)*perc+min
    under = (matrix < thr).sum()
    if verbose:  
        print("currently under: ", (under*100)//tot, "%")
    
    done = False
    STEP = 0.01
    MULT = 1.2
    SENS = 0.01
    sens = 0.01
    step = 0.01
    mult = 0.5
    
    dir = 1 #aumento la soglia
    dir_count=0
    switch_dir = 0
    count = 0
    
    if under/tot - perc > SENS:
            dir = -1
    
    
    while not done:
        
        
        if abs(under/tot - perc) < sens:
            done = True
            continue
        
        thr = thr + (step*dir)
        under = (matrix < thr).sum()
        
        tmpdir=1
        if under/tot - perc > 0:
            tmpdir = -1

        if tmpdir==dir:
            dir_count+=1
            switch_dir=0
        else:
            dir *=-1
            dir_count=0
            mult=MULT
            step = STEP
            switch_dir+=1

        if switch_dir>2:
            sens*=1.1
        
        if dir_count>2:
            step=STEP*mult*(dir_count-2)
        
        if verbose:
            print("count: ", count)
            print("thr: ",thr)
            print("step:",step)
            print("mult: ", mult)
            print("currently under: ", (under*100)//tot, "%")
            print("dir: ", dir)    
            print("dir_count: ", dir_count)
            print("switch_dir: ", switch_dir)
            print("sens: ", sens)
            print("under/tot - perc",under/tot - perc)
            print("-"*20)
            tmp = matrix.copy()
            tmp[matrix<thr] = thr
            plot_3d_noisemap(tmp) 
        count+=1
  
        
    return thr, count

@timeit
def benchmark(noise_map):
    max = 0
    for i in range(100):
        fr = random.random()
        thr, count = flood(noise_map, fr )
        print(count)
        if count>max:
            max = count
    
    print("max: ", max)
        
def main():
    
    BIG = 0
    # Example usage:
    width = 200
    height = 200
    scale = 200
    octaves = 6 #irregolarità larga scala
    persistence = 0.5   #densità picchi
    lacunarity = 2.0   #defferenza di altezza tra i picchi
    random.seed(datetime.datetime.now().timestamp())
    seed = random.randint(0,100000000)
    
    if BIG:
        width = 2000
        height = 2000
        scale = 1000
        octaves = 6 #irregolarità larga scala
        persistence = 0.5   #densità picchi
        lacunarity = 2.0   #defferenza di altezza tra i picchi
        random.seed(datetime.datetime.now().timestamp())
        seed = random.randint(0,100000000)
    
    
    
    print("Generating Noise Map")
    print("seed: ", seed)
    noise_map = generate_noise_map(width, height, scale, octaves, persistence, lacunarity, seed)
    print("done")
    # benchmark(noise_map)
    # print(noise_map)
    
    # plot_3d_noisemap(np.abs(noise_map)) 
    
    # noise_map=np.abs(noise_map)
    
    # plot_3d_noisemap(noise_map)
    
    # print("flooding")
    # thr, count = flood(noise_map, 0.3 )
    # print(f"flooded in {count} steps")
    # noise_map[noise_map<thr] = thr
    
    # # save_noise_map_as_png(noise_map=noise_map, filename="1.png")
    
    # # plot(noise_map)
    
    # plot_3d_noisemap(noise_map)
    
    # max = np.max(noise_map)
    # min = np.min(noise_map)
    # mean = np.mean(noise_map)
    
    # noise_map=noise_map-mean
    
    # noise_map=np.abs(noise_map)
    
    # plot_3d_noisemap(noise_map) 
    # noise_map=1-noise_map
    # noise_map[noise_map<-0.1] = -0.1
    # plot_3d_noisemap(noise_map)
    

    # min = np.min(noise_map) 
    # max = np.max(noise_map) 
    
    # avg =( max -min ) /2 
    
    # print("min + avg: ", min+avg)
    # noise_map[noise_map < (avg +min)] = noise_map[noise_map < (avg + min)] - (noise_map[noise_map < (avg + min)] - avg - min)*2 
    # # noise_map[noise_map >= (avg +min)] = 0
    # noise_map=hemisphere(height,width)
    noise_map[0][0]=-1
    noise_map[0][1]=1
    
    plot_3d_noisemap(noise_map)

    # exit(0)
    
    cratered = crater(noise_map.copy(), (70,90), 50, 0)
    
    n_crater = 20
    
    centers= []
    
    while n_crater>0:
        radius = random.randint(5, 20)
        posx = random.randint(radius, width-radius*2-1)
        posy = random.randint(radius, height-radius*2-1)
        intensity = radius/150
        centers.append((posx,posy, radius))
        cratered = crater(cratered, (posx,posy), radius, intensity)
        n_crater-=1
        print("craters left: ",n_crater)
        # cratered[][]
        if n_crater%5==0:
            bak = []
            for c in centers:
                bak.append(cratered[ c[0]+ c[2] ][ c[1]+ c[2] ])    #salvo tutti i valori dei crateri
                cratered[ c[0]+ c[2] ][ c[1]+ c[2] ]=1  #sovrascrivo 
            plot_3d_noisemap(cratered)
            # time.sleep(1)
            for b in bak:
                cratered[ c[0]+ c[2] ][ c[1]+ c[2] ]=b  #ripristino
                        
    
    # plot_3d_noisemap(cratered)

def test():
    # Example usage:
    width = 200
    height = 200
    scale = 100
    octaves = 4 #irregolarità larga scala
    persistence = 0.5   #densità picchi
    lacunarity = 2.0
    seed = 42

    noise_map = generate_noise_map(width, height, scale, octaves, persistence, lacunarity, seed)
    # print(simplex_noise)
    
    cr = crater(noise_map, (80,80), 50, 1)
    print(cr.shape)
    plot_3d_noisemap(cr)
    save_noise_map_as_png(cr, "crater.png")  
    
    print(cr)
    
    
def crater(matrix, pos, radius, intensity=1):
    x=pos[0]
    y=pos[1]
    
    crater = hemisphere(radius*2,radius*2) / radius
    
    center = (radius , radius )
    
    # for i in range(radius*2):
    #     for j in range(radius*2):
    #         # crater[i][j]= math.dist(center, (i,j))
            
    #         x_rad = (i - center[0])/center[0]
    #         y_rad = (j - center[1])/(center[1])
                       
    #         sum_tmp = (1-x_rad**2 -y_rad**2)
    #         if sum_tmp<0:
    #              crater[i][j]=0
    #         else:
    #             crater[i][j]= np.sqrt(sum_tmp)

    
    crater = crater * intensity
     
    
    matrix[x:(x+radius*2) , y:(y+radius*2)]  = matrix[x:(x+radius*2) , y:(y+radius*2)] - (crater) 
    # matrix[x:(x+radius*2) , y:(y+radius*2)]  = 1-  crater
    
    
    
    return matrix
    
if __name__ == "__main__":
    main()
    # test()