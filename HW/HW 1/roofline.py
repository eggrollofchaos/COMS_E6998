# wax1
import matplotlib.pyplot as plt
import numpy as np

"""
Q2: Roofline Model Generator.
This script plots the peak performance boundaries and 
placeholder points for your 10 benchmark configurations.
"""

def plot_roofline(peak_gflops=200, peak_bw_gbs=30):
    # Arithmetic Intensity (FLOPs / Byte)
    # For dot product: 2 FLOPs / (2 * 4 bytes) = 0.25 FLOP/byte
    ai_dot_product = 0.25

    results = {
        "C1 Small": 2.126,  # Example: Change to your actual value
        "C1 Large": 2.004,
        "C2 Small": 5.817,
        "C2 Large": 3.978,
        "C3 Small": 26.77,
        "C3 Large": 14.010,
        "C4 Small": 0.0102,
        "C4 Large": 0.0102,
        "C5 Small": 12.890,
        "C5 Large": 3.939
    }
    
    # Define range for Arithmetric Intensity (x-axis)
    ai_range = np.logspace(-2, 2, 100)
    
    # Calculate Roofline: min(Peak GFLOP/s, Peak BW * AI)
    roofline = np.minimum(peak_gflops, peak_bw_gbs * ai_range)
    
    plt.figure(figsize=(10, 7))

    # Plot the "Assignment" Roofline
    plt.loglog(ai_range, roofline, label='Theoretical Roofline (30 GB/s)', color='red', lw=3)

    # Plot a "Cache-Aware" Roofline? 
    # If your C3 Small is at 28 GFLOPS, your cache bandwidth is at least 28/0.25 = 112 GB/s
    cache_bw = 120 
    cache_roofline = np.minimum(peak_gflops, cache_bw * ai_range)
    plt.loglog(ai_range, cache_roofline, '--', label='Potential Cache Ceiling', color='orange', alpha=0.5)

    # Plot actual points
    colors = plt.cm.get_cmap('tab10', len(results))
    for i, (label, gflops) in enumerate(results.items()):
        plt.scatter(ai_dot_product, gflops, label=f"{label}: {gflops} GFLOPS", s=100, edgecolors='black', zorder=5)
    
    # Vertical line for Dot Product Arithmetic Intensity
    # plt.axvline(x=ai_dot_product, color='blue', linestyle='--', label=f'AI = {ai_dot_product}')
    plt.axvline(x=ai_dot_product, color='blue', linestyle=':', alpha=0.6) #, label=f'AI = {ai_dot_product}')
    plt.text(ai_dot_product, 0.0006, f' AI={ai_dot_product}', color='blue', fontweight='bold')
    
    plt.xlabel('Arithmetic Intensity (FLOP/Byte)')
    plt.ylabel('Performance (GFLOP/s)')
    plt.title('Roofline Model for given specs: Peak 200 GFLOP/s, 30 GB/s BW')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    plt.savefig('roofline_model2.png')
    print("Roofline model saved as roofline_model2.png")
    plt.show()

if __name__ == "__main__":
    plot_roofline()