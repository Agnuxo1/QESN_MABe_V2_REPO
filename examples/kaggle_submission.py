#!/usr/bin/env python3
"""
QESN-MABe V2: Kaggle Submission Script
Author: Francisco Angulo de Lafuente
License: MIT

This script demonstrates how to use QESN for Kaggle MABe 2022 submission.
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import sys
import os
from typing import List, Tuple, Dict
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from python.qesn_inference import QESNInference
    QESN_AVAILABLE = True
except ImportError:
    QESN_AVAILABLE = False
    print("⚠️  QESN inference module not available. Using simulation mode.")

class KaggleSubmission:
    """Kaggle submission handler for QESN-MABe V2"""
    
    def __init__(self, model_path: str = None, config_path: str = None):
        """
        Initialize Kaggle submission
        
        Args:
            model_path: Path to model_weights.bin
            config_path: Path to model_config.json
        """
        self.behaviors = [
            "allogroom", "approach", "attack", "attemptmount", "avoid",
            "biteobject", "chase", "chaseattack", "climb", "defend",
            "dig", "disengage", "dominance", "dominancegroom", "dominancemount",
            "ejaculate", "escape", "exploreobject", "flinch", "follow",
            "freeze", "genitalgroom", "huddle", "intromit", "mount",
            "rear", "reciprocalsniff", "rest", "run", "selfgroom",
            "shepherd", "sniff", "sniffbody", "sniffface", "sniffgenital",
            "submit", "tussle"
        ]
        
        if QESN_AVAILABLE and model_path and config_path:
            self.model = QESNInference(model_path, config_path)
            print(f"✅ QESN model loaded from {model_path}")
        else:
            self.model = None
            print("⚠️  Using simulation mode (no trained model)")
    
    def load_test_metadata(self, metadata_path: str) -> pd.DataFrame:
        """Load test metadata CSV"""
        try:
            metadata = pd.read_csv(metadata_path)
            print(f"✅ Loaded metadata: {len(metadata)} videos")
            return metadata
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            return None
    
    def load_tracking_data(self, tracking_path: str) -> pd.DataFrame:
        """Load tracking data from parquet file"""
        try:
            table = pq.read_table(tracking_path)
            df = table.to_pandas()
            print(f"✅ Loaded tracking data: {len(df)} rows")
            return df
        except Exception as e:
            print(f"❌ Error loading tracking data: {e}")
            return None
    
    def convert_tracking_to_keypoints(self, tracking_df: pd.DataFrame, 
                                    video_width: int, video_height: int) -> np.ndarray:
        """
        Convert tracking DataFrame to keypoints array
        
        Args:
            tracking_df: DataFrame with columns [video_frame, mouse_id, bodypart, x, y, likelihood]
            video_width: Video width in pixels
            video_height: Video height in pixels
            
        Returns:
            keypoints: (num_frames, num_mice, num_keypoints, 3) array
        """
        # Get unique frames and mice
        frames = sorted(tracking_df['video_frame'].unique())
        mice = sorted(tracking_df['mouse_id'].unique())
        bodyparts = sorted(tracking_df['bodypart'].unique())
        
        # Initialize keypoints array
        num_frames = len(frames)
        num_mice = len(mice)
        num_keypoints = len(bodyparts)
        
        keypoints = np.full((num_frames, num_mice, num_keypoints, 3), np.nan)
        
        # Fill keypoints data
        for _, row in tracking_df.iterrows():
            frame_idx = frames.index(row['video_frame'])
            mouse_idx = mice.index(row['mouse_id'])
            kp_idx = bodyparts.index(row['bodypart'])
            
            # Normalize coordinates
            x = row['x'] / video_width
            y = row['y'] / video_height
            conf = row['likelihood']
            
            keypoints[frame_idx, mouse_idx, kp_idx, 0] = x * video_width
            keypoints[frame_idx, mouse_idx, kp_idx, 1] = y * video_height
            keypoints[frame_idx, mouse_idx, kp_idx, 2] = conf
        
        return keypoints
    
    def generate_windows(self, keypoints: np.ndarray, window_size: int = 30, 
                        stride: int = 15) -> List[np.ndarray]:
        """Generate sliding windows from keypoints"""
        windows = []
        
        for start_frame in range(0, len(keypoints) - window_size + 1, stride):
            window = keypoints[start_frame:start_frame + window_size]
            windows.append(window)
        
        return windows
    
    def predict_window(self, keypoints: np.ndarray, video_width: int, 
                      video_height: int) -> Tuple[int, np.ndarray, str]:
        """Predict behavior for a single window"""
        
        if self.model:
            # Use real QESN model
            pred_idx, probs, pred_name = self.model.predict(
                keypoints, video_width, video_height
            )
        else:
            # Simulation mode
            pred_idx, probs, pred_name = self._simulate_prediction(keypoints)
        
        return pred_idx, probs, pred_name
    
    def _simulate_prediction(self, keypoints: np.ndarray) -> Tuple[int, np.ndarray, str]:
        """Simulate prediction for demo purposes"""
        # Simple simulation based on keypoint movement patterns
        movement = np.nanmean(np.diff(keypoints, axis=0), axis=(1, 2, 3))
        avg_movement = np.nanmean(movement)
        
        # Map movement to behavior (simplified)
        if avg_movement > 10:
            pred_name = "run"
        elif avg_movement > 5:
            pred_name = "approach"
        elif avg_movement > 2:
            pred_name = "sniff"
        else:
            pred_name = "rest"
        
        pred_idx = self.behaviors.index(pred_name)
        
        # Generate realistic probabilities
        probs = np.random.exponential(0.1, len(self.behaviors))
        probs[pred_idx] *= 5  # Boost predicted class
        probs = probs / probs.sum()
        
        return pred_idx, probs, pred_name
    
    def process_video(self, video_id: str, tracking_path: str, 
                    video_width: int, video_height: int) -> List[Dict]:
        """Process a single video and generate predictions"""
        
        print(f"🎬 Processing video: {video_id}")
        
        # Load tracking data
        tracking_df = self.load_tracking_data(tracking_path)
        if tracking_df is None:
            return []
        
        # Convert to keypoints
        keypoints = self.convert_tracking_to_keypoints(
            tracking_df, video_width, video_height
        )
        
        # Generate windows
        windows = self.generate_windows(keypoints, window_size=30, stride=15)
        
        # Predict for each window
        results = []
        for i, window in enumerate(windows):
            start_frame = i * 15
            end_frame = start_frame + 30
            
            pred_idx, probs, pred_name = self.predict_window(
                window, video_width, video_height
            )
            
            results.append({
                'video_id': video_id,
                'frame_start': start_frame,
                'frame_end': end_frame,
                'prediction': pred_name,
                'confidence': probs[pred_idx],
                'agent_id': 0,  # Default agent
                'target_id': 1  # Default target
            })
        
        print(f"✅ Generated {len(results)} predictions for {video_id}")
        return results
    
    def create_submission(self, test_metadata_path: str, 
                         tracking_dir: str, output_path: str = "submission.csv"):
        """Create complete Kaggle submission"""
        
        print("🚀 Starting Kaggle submission generation...")
        print("=" * 60)
        
        # Load test metadata
        metadata = self.load_test_metadata(test_metadata_path)
        if metadata is None:
            return
        
        all_results = []
        total_videos = len(metadata)
        
        for idx, row in metadata.iterrows():
            video_id = row['video_id']
            video_width = row['width']
            video_height = row['height']
            
            # Construct tracking path
            tracking_path = os.path.join(tracking_dir, f"{video_id}.parquet")
            
            if not os.path.exists(tracking_path):
                print(f"⚠️  Tracking file not found: {tracking_path}")
                continue
            
            # Process video
            video_results = self.process_video(
                video_id, tracking_path, video_width, video_height
            )
            all_results.extend(video_results)
            
            # Progress update
            if (idx + 1) % 10 == 0:
                print(f"📊 Progress: {idx + 1}/{total_videos} videos processed")
        
        # Create submission DataFrame
        submission_df = pd.DataFrame(all_results)
        
        # Save submission
        submission_df.to_csv(output_path, index=False)
        
        print(f"\\n✅ Submission created: {output_path}")
        print(f"📊 Total predictions: {len(submission_df)}")
        print(f"🎯 Unique behaviors: {submission_df['prediction'].nunique()}")
        print(f"📈 Average confidence: {submission_df['confidence'].mean():.3f}")
        
        # Show prediction distribution
        print("\\n📊 Prediction distribution:")
        pred_counts = submission_df['prediction'].value_counts().head(10)
        for behavior, count in pred_counts.items():
            print(f"  {behavior}: {count} ({count/len(submission_df)*100:.1f}%)")
        
        return submission_df

def main():
    """Main function for Kaggle submission"""
    
    print("🏆 QESN-MABe V2: Kaggle Submission")
    print("=" * 50)
    print("Author: Francisco Angulo de Lafuente")
    print("GitHub: https://github.com/Agnuxo1")
    print("=" * 50)
    
    # Initialize submission handler
    submission = KaggleSubmission()
    
    # Example usage (adapt paths to your Kaggle environment)
    test_metadata_path = "/kaggle/input/mabe-2022/test.csv"
    tracking_dir = "/kaggle/input/mabe-2022/test_tracking"
    output_path = "submission.csv"
    
    # Check if running in Kaggle environment
    if os.path.exists("/kaggle"):
        print("🎯 Running in Kaggle environment")
        submission.create_submission(test_metadata_path, tracking_dir, output_path)
    else:
        print("💻 Running locally - creating example submission")
        
        # Create example submission with simulated data
        example_results = []
        for i in range(100):  # Simulate 100 predictions
            behavior = np.random.choice(submission.behaviors)
            confidence = np.random.uniform(0.3, 0.9)
            
            example_results.append({
                'video_id': f'example_video_{i//10:03d}',
                'frame_start': (i % 10) * 15,
                'frame_end': (i % 10) * 15 + 30,
                'prediction': behavior,
                'confidence': confidence,
                'agent_id': 0,
                'target_id': 1
            })
        
        example_df = pd.DataFrame(example_results)
        example_df.to_csv("example_submission.csv", index=False)
        
        print("✅ Example submission created: example_submission.csv")
        print(f"📊 Example predictions: {len(example_df)}")
    
    print("\\n📚 For more information:")
    print("   - GitHub: https://github.com/Agnuxo1/QESN-MABe-V2")
    print("   - Kaggle: https://www.kaggle.com/franciscoangulo")
    print("   - Documentation: docs/")

if __name__ == "__main__":
    main()
