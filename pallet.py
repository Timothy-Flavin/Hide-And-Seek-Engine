import argparse
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import traceback

def rgb_to_hex(rgb):
    """Converts an RGB tuple/array to a HEX string."""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Extract a color palette from the FULL image using KMeans, then save a downsized quantized result.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image file")
    parser.add_argument('--K', type=int, default=5, help="Number of colors to extract for the palette (default: 5)")
    parser.add_argument('--N', type=int, default=100, help="Downsize image to NxN after clustering (default: 100)")
    args = parser.parse_args()

    try:
        # 1. Load the full image
        print(f"Loading image from '{args.image_path}'...")
        img = Image.open(args.image_path).convert('RGB')
        
        # 2. Extract pixels from the FULL resolution image
        print("Extracting pixels from the full resolution image...")
        full_img_array = np.array(img)
        full_pixels = full_img_array.reshape(-1, 3)
        
        # 3. Use KMeans to cluster the colors on the FULL image
        print(f"Clustering into {args.K} colors using KMeans...")
        print("(Note: Processing the full resolution image may take a moment)")
        kmeans = KMeans(n_clusters=args.K, random_state=42, n_init='auto')
        kmeans.fit(full_pixels)
        
        # Extract the dominant colors (the cluster centers)
        colors = kmeans.cluster_centers_.astype(np.uint8)
        full_labels = kmeans.labels_ # We use this to calculate the % representation
        
        # 4. Downsize the original image to NxN
        print(f"Resizing image to {args.N}x{args.N}...")
        resized_img = img.resize((args.N, args.N))
        
        # Convert the resized image to a 2D array of pixels
        resized_pixels = np.array(resized_img).reshape(-1, 3)
        
        # 5. Predict the nearest color cluster for the RESIZED pixels
        print("Mapping resized image pixels to the extracted palette...")
        resized_labels = kmeans.predict(resized_pixels)
        
        # Map each resized pixel to its corresponding cluster center color
        quantized_pixels = colors[resized_labels]
        
        # Reshape the flat array back into the 2D image shape (N, N, 3)
        quantized_image_array = quantized_pixels.reshape(args.N, args.N, 3)
        
        # Convert the numpy array back into a Pillow Image
        quantized_img = Image.fromarray(quantized_image_array)
        
        # Save the new quantized image
        output_filename = "newlevel.png"
        quantized_img.save(output_filename)
        print(f"\nSaved compressed KMeans image to '{output_filename}'")
        
        # --- Extracted Color Palette (Based on the Full Image) ---
        counts = np.bincount(full_labels)
        sorted_indices = np.argsort(counts)[::-1] # Sort descending
        
        print("\n--- Extracted Color Palette ---")
        for i, idx in enumerate(sorted_indices):
            color = colors[idx]
            hex_color = rgb_to_hex(color)
            percentage = (counts[idx] / len(full_labels)) * 100
            print(f"Color {i+1}: RGB {tuple(color)} | HEX {hex_color} | Representation: {percentage:.2f}%")
            
    except FileNotFoundError:
        print(f"Error: Could not find an image at '{args.image_path}'. Please check the path.")
    except Exception as e:
        traceback.print_exc()
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()