"""Utility functions for video processing."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_frames(video_path: str) -> Tuple[List[np.ndarray], dict]:
    """
    Extract all frames from a video file.
    
    Args:
        video_path: Path to input video file
        
    Returns:
        Tuple of (frames list, video_info dict)
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    video_info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    
    frames = []
    logger.info(f"Extracting frames from {video_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    logger.info(f"✓ Extracted {len(frames)} frames")
    logger.info(f"  Resolution: {video_info['width']}x{video_info['height']}")
    logger.info(f"  FPS: {video_info['fps']}")
    
    return frames, video_info


def write_video(frames: List[np.ndarray], 
                output_path: str, 
                fps: float = 30.0,
                codec: str = 'mp4v') -> None:
    """
    Write frames to a video file.
    
    Args:
        frames: List of video frames (numpy arrays)
        output_path: Output file path
        fps: Frames per second
        codec: Video codec (default: mp4v)
    """
    if not frames:
        raise ValueError("No frames to write")
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Could not open video writer: {output_path}")
    
    logger.info(f"Writing {len(frames)} frames to {output_path}...")
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    logger.info(f"✓ Video saved to {output_path}")


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize frame maintaining aspect ratio.
    
    Args:
        frame: Input frame
        width: Target width
        height: Target height
        
    Returns:
        Resized frame
    """
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def save_execution_log(log_path: str, 
                       execution_time: float,
                       frame_count: int,
                       video_info: dict,
                       additional_info: dict = None) -> None:
    """
    Save execution log with performance metrics.
    
    Args:
        log_path: Path to save log file
        execution_time: Total execution time in seconds
        frame_count: Number of frames processed
        video_info: Video metadata
        additional_info: Additional metrics to log
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("VIDEO FRAME RECONSTRUCTION - EXECUTION LOG\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("VIDEO INFORMATION:\n")
        f.write(f"  Frames: {frame_count}\n")
        f.write(f"  Resolution: {video_info['width']}x{video_info['height']}\n")
        f.write(f"  FPS: {video_info['fps']}\n")
        f.write(f"  Duration: {frame_count / video_info['fps']:.2f}s\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  Total Execution Time: {execution_time:.2f} seconds\n")
        f.write(f"  Time per Frame: {execution_time / frame_count:.4f}s\n")
        f.write(f"  Processing Speed: {frame_count / execution_time:.2f} FPS\n\n")
        
        if additional_info:
            f.write("ADDITIONAL METRICS:\n")
            for key, value in additional_info.items():
                f.write(f"  {key}: {value}\n")
    
    logger.info(f"✓ Execution log saved to {log_path}")
