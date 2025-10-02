#!/usr/bin/env python3
"""
QESN-MABe V2: Parquet Preprocessor
Author: Francisco Angulo de Lafuente

Converts parquet files to simplified binary format for C++ trainer.
This bypasses the need for Apache Arrow C++ library.
"""

import os
import sys
import struct
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Dataset paths
TRAIN_CSV = r"E:\QESN-MABe\train.csv"
TRACKING_ROOT = r"E:\QESN-MABe\train_tracking"
ANNOTATION_ROOT = r"E:\QESN-MABe\train_annotation"
OUTPUT_ROOT = r"E:\QESN_MABe_V2\data\preprocessed"

# 37 MABe action names
MABE_ACTION_NAMES = [
    "allogroom", "approach", "attack", "attemptmount", "avoid",
    "biteobject", "chase", "chaseattack", "climb", "defend",
    "dig", "disengage", "dominance", "dominancegroom", "dominancemount",
    "ejaculate", "escape", "exploreobject", "flinch", "follow",
    "freeze", "genitalgroom", "huddle", "intromit", "mount",
    "rear", "reciprocalsniff", "rest", "run", "selfgroom",
    "shepherd", "sniff", "sniffbody", "sniffface", "sniffgenital",
    "submit", "tussle"
]

ACTION_TO_INDEX = {name: idx for idx, name in enumerate(MABE_ACTION_NAMES)}


def load_parquet_sequence(video_id, lab_id, width, height, fps):
    """Load and convert one parquet sequence"""

    # Paths
    tracking_path = os.path.join(TRACKING_ROOT, lab_id, f"{video_id}.parquet")
    annotation_path = os.path.join(ANNOTATION_ROOT, lab_id, f"{video_id}.csv")

    # Load tracking
    if not os.path.exists(tracking_path):
        print(f"  WARNING: Tracking file not found: {tracking_path}")
        return None

    df_tracking = pd.read_parquet(tracking_path)

    # Get unique values
    frames = sorted(df_tracking['video_frame'].unique())
    bodyparts = sorted(df_tracking['bodypart'].unique())
    mice = sorted(df_tracking['mouse_id'].unique())

    num_frames = len(frames)
    num_mice = len(mice)
    num_keypoints = len(bodyparts)

    print(f"  Frames: {num_frames}, Mice: {num_mice}, Keypoints: {num_keypoints}")

    # Build keypoint array: [frames][mice][keypoints][x,y,confidence]
    keypoints = np.zeros((num_frames, num_mice, num_keypoints, 3), dtype=np.float32)

    for frame_idx, frame in enumerate(frames):
        frame_data = df_tracking[df_tracking['video_frame'] == frame]

        for mouse_idx, mouse_id in enumerate(mice):
            mouse_data = frame_data[frame_data['mouse_id'] == mouse_id]

            for kp_idx, bodypart in enumerate(bodyparts):
                kp_data = mouse_data[mouse_data['bodypart'] == bodypart]

                if len(kp_data) > 0:
                    # Normalize by video dimensions
                    x = kp_data['x'].values[0] / width
                    y = kp_data['y'].values[0] / height
                    conf = kp_data['likelihood'].values[0]

                    keypoints[frame_idx, mouse_idx, kp_idx] = [x, y, conf]
                else:
                    keypoints[frame_idx, mouse_idx, kp_idx] = [np.nan, np.nan, 0.0]

    # Load annotations
    labels = {}
    if os.path.exists(annotation_path):
        df_annot = pd.read_csv(annotation_path)

        for _, row in df_annot.iterrows():
            action = row['action']
            if action not in ACTION_TO_INDEX:
                continue

            action_idx = ACTION_TO_INDEX[action]
            start_frame = int(row['start_frame'])
            stop_frame = int(row['stop_frame'])

            # Assign label to all frames in range
            for frame in range(start_frame, stop_frame):
                if frame < num_frames:
                    labels[frame] = action_idx

    return {
        'video_id': video_id,
        'lab_id': lab_id,
        'width': width,
        'height': height,
        'fps': fps,
        'keypoints': keypoints,
        'labels': labels,
        'frames': frames
    }


def save_binary_sequence(sequence, output_path):
    """Save sequence in binary format for C++"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        # Header
        video_id_bytes = sequence['video_id'].encode('utf-8')
        lab_id_bytes = sequence['lab_id'].encode('utf-8')

        f.write(struct.pack('I', len(video_id_bytes)))
        f.write(video_id_bytes)
        f.write(struct.pack('I', len(lab_id_bytes)))
        f.write(lab_id_bytes)

        # Metadata
        f.write(struct.pack('i', sequence['width']))
        f.write(struct.pack('i', sequence['height']))
        f.write(struct.pack('d', sequence['fps']))

        # Dimensions
        keypoints = sequence['keypoints']
        num_frames, num_mice, num_keypoints, _ = keypoints.shape

        f.write(struct.pack('I', num_frames))
        f.write(struct.pack('I', num_mice))
        f.write(struct.pack('I', num_keypoints))

        # Keypoints (flattened)
        keypoints_flat = keypoints.flatten().astype(np.float32)
        f.write(keypoints_flat.tobytes())

        # Labels
        labels = sequence['labels']
        f.write(struct.pack('I', len(labels)))

        for frame_idx, action_idx in sorted(labels.items()):
            f.write(struct.pack('I', frame_idx))
            f.write(struct.pack('i', action_idx))

    print(f"  Saved: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")


def main():
    print("=" * 60)
    print("QESN-MABe V2: Parquet Preprocessor")
    print("=" * 60)
    print()

    # Check paths
    if not os.path.exists(TRAIN_CSV):
        print(f"ERROR: Train CSV not found: {TRAIN_CSV}")
        return 1

    if not os.path.exists(TRACKING_ROOT):
        print(f"ERROR: Tracking root not found: {TRACKING_ROOT}")
        return 1

    # Create output directory
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Read metadata
    print(f"Reading metadata: {TRAIN_CSV}")
    df_train = pd.read_csv(TRAIN_CSV)
    print(f"Found {len(df_train)} sequences\n")

    # Ask how many to process
    max_sequences = input(f"How many sequences to process? (Enter for all, or number): ").strip()
    if max_sequences:
        max_sequences = int(max_sequences)
        df_train = df_train.head(max_sequences)

    print(f"\nProcessing {len(df_train)} sequences...\n")

    # Process each sequence
    success_count = 0
    fail_count = 0

    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Converting"):
        video_id = row['video_id']
        lab_id = row['lab_id']
        width = int(row['video_width_pix'])
        height = int(row['video_height_pix'])
        fps = float(row['fps'])

        print(f"\n[{idx + 1}/{len(df_train)}] {video_id} ({width}x{height})")

        try:
            sequence = load_parquet_sequence(video_id, lab_id, width, height, fps)

            if sequence is None:
                fail_count += 1
                continue

            output_path = os.path.join(OUTPUT_ROOT, lab_id, f"{video_id}.bin")
            save_binary_sequence(sequence, output_path)

            success_count += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            fail_count += 1

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output: {OUTPUT_ROOT}")
    print()

    # Create manifest file
    manifest_path = os.path.join(OUTPUT_ROOT, "manifest.txt")
    with open(manifest_path, 'w') as f:
        for root, dirs, files in os.walk(OUTPUT_ROOT):
            for file in files:
                if file.endswith('.bin'):
                    rel_path = os.path.relpath(os.path.join(root, file), OUTPUT_ROOT)
                    f.write(rel_path + '\n')

    print(f"Created manifest: {manifest_path}")
    print(f"Total files: {success_count}")
    print()
    print("Next step: Run scripts\\build.bat to compile C++ trainer")

    return 0


if __name__ == '__main__':
    sys.exit(main())
