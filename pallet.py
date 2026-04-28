import argparse
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def rgb_to_hex(rgb):
    """Converts an RGB tuple/array to a HEX string."""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Extract a color palette from an image using KMeans.")
    parser.add_argument('image_path', type=str, help="Path to the input image file")
    parser.add_argument('--K', type=int, default=5, help="Number of colors to extract for the palette (default: 5)")
    parser.add_argument('--N', type=int, default=100, help="Downsize image to NxN before clustering (default: 100)")
    args = parser.parse_args()

    try:
        # 1. Load the image and force it to RGB (to drop alpha channels if they exist)
        print(f"Loading image from '{args.image_path}'...")
        img = Image.open(args.image_path).convert('RGB')
        
        # 2. Downsize the image to NxN to speed up KMeans computation
        print(f"Resizing image to {args.N}x{args.N}...")
        img = img.resize((args.N, args.N))
        
        # 3. Convert image to a numpy array and reshape it into a 2D array of pixels
        # img_array shape is (N, N, 3), reshaping it to (N*N, 3)
        img_array = np.array(img)
        pixels = img_array.reshape(-1, 3)
        
        # 4. Use KMeans to cluster the colors
        print(f"Clustering into {args.K} colors using KMeans...")
        # n_init='auto' suppresses an sklearn warning about default behavior changes
        kmeans = KMeans(n_clusters=args.K, random_state=42, n_init='auto')
        kmeans.fit(pixels)
        
        # 5. Extract the dominant colors (the cluster centers)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Sort colors by how many pixels belong to each cluster (optional, but helpful)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        sorted_indices = np.argsort(counts)[::-1] # Sort descending
        
        print("\n--- Extracted Color Palette ---")
        for i, idx in enumerate(sorted_indices):
            color = colors[idx]
            hex_color = rgb_to_hex(color)
            percentage = (counts[idx] / len(labels)) * 100
            print(f"Color {i+1}: RGB {tuple(color):<15} | HEX {hex_color:<8} | Representation: {percentage:.2f}%")
            
    except FileNotFoundError:
        print(f"Error: Could not find an image at '{args.image_path}'. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()