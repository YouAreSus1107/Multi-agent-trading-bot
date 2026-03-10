import pandas as pd
import os

def generate_curated_models():
    """Generates a CSV of highly vetted model configurations, focusing on the new MTF Trailing SL."""
    
    models = [
        {
            "category": "MTF Trend Follower (Loose Entry)",
            "long_hybrid": 65, "long_exec": 30, "long_vwap": 0.5,
            "short_hybrid": 35, "short_exec": 30, "short_vwap": -0.5,
            "stop_r": 1.5, "target_r": 3.0, 
            "trail_1r": False, "trail_2r": False, 
            "trail_activation_r": 0.0, "trailing_distance": 0.0,
            "use_mtf": True
        },
        {
            "category": "MTF Strict Execution (High Conviction)",
            "long_hybrid": 70, "long_exec": 45, "long_vwap": -1.0,
            "short_hybrid": 30, "short_exec": 45, "short_vwap": 1.0,
            "stop_r": 1.5, "target_r": 3.0, 
            "trail_1r": False, "trail_2r": False, 
            "trail_activation_r": 0.0, "trailing_distance": 0.0,
            "use_mtf": True
        },
        {
            "category": "MTF Pullback Buyer (VWAP Bounce)",
            "long_hybrid": 60, "long_exec": 50, "long_vwap": -2.0,
            "short_hybrid": 40, "short_exec": 50, "short_vwap": 2.0,
            "stop_r": 1.5, "target_r": 3.0, 
            "trail_1r": False, "trail_2r": False, 
            "trail_activation_r": 0.0, "trailing_distance": 0.0,
            "use_mtf": True
        },
        {
            "category": "Classic Optimized (No Trailing, Fixed Target)",
            "long_hybrid": 72, "long_exec": 35, "long_vwap": -1.0,
            "short_hybrid": 28, "short_exec": 35, "short_vwap": 1.0,
            "stop_r": 1.25, "target_r": 2.5, 
            "trail_1r": False, "trail_2r": False, 
            "trail_activation_r": 0.0, "trailing_distance": 0.0,
            "use_mtf": False
        },
        {
            "category": "Classic Runner (Standard Breakeven Trail)",
            "long_hybrid": 68, "long_exec": 40, "long_vwap": 0.0,
            "short_hybrid": 32, "short_exec": 40, "short_vwap": 0.0,
            "stop_r": 1.5, "target_r": 4.0, 
            "trail_1r": True, "trail_2r": True, 
            "trail_activation_r": 1.5, "trailing_distance": 2.0,
            "use_mtf": False
        }
    ]

    df = pd.DataFrame(models)
    
    # Save the file
    out_path = os.path.join(os.path.dirname(__file__), "selected_models.csv")
    df.to_csv(out_path, index=False)
    
    print(f"Generated {len(models)} curated models -> {os.path.basename(out_path)}")

if __name__ == "__main__":
    generate_curated_models()
