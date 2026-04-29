import os
import subprocess
import tempfile
import cv2
import numpy as np
from pathlib import Path

# Try to get FFmpeg path from portable-ffmpeg
try:
    from portable_ffmpeg import get_ffmpeg
    FFMPEG_PATH, FFPROBE_PATH = get_ffmpeg()
    print(f"✅ portable-ffmpeg found at: {FFMPEG_PATH}")
    HAS_FFMPEG = True
except ImportError:
    print("⚠️ portable-ffmpeg not installed, trying system FFmpeg")
    # Try to find ffmpeg in system PATH
    import shutil
    FFMPEG_PATH = shutil.which('ffmpeg')
    if FFMPEG_PATH:
        print(f"✅ System FFmpeg found at: {FFMPEG_PATH}")
        HAS_FFMPEG = True
    else:
        print("❌ No FFmpeg found anywhere. Will use OpenCV fallback only.")
        HAS_FFMPEG = False
        FFMPEG_PATH = 'ffmpeg'  # Will fail, but we'll catch it

def save_video_with_ffmpeg(
    frames: list,
    output_path: str,
    fps: int = 20,
    quality: str = "slow",#medium
    for_web: bool = True
) -> bool:
    """
    Save video using FFmpeg for optimal web playback
    """
    if not HAS_FFMPEG:
        print("⚠️ FFmpeg not available, use fallback instead")
        return False
        
    try:
        if not frames:
            print("❌ No frames to save")
            return False
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Save frames as images
            print(f"📸 Saving {len(frames)} frames as images...")
            for i, frame in enumerate(frames):
                if isinstance(frame, np.ndarray):
                    # Ensure correct color format
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame_rgb = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = frame
                    
                    frame_path = temp_dir_path / f"frame_{i:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Get first frame dimensions
            height, width = frames[0].shape[:2]
            
            # Ensure dimensions are even (required for H.264)
            if height % 2 != 0:
                height += 1
            if width % 2 != 0:
                width += 1
            
            # Build FFmpeg command
            cmd = [
                FFMPEG_PATH, '-y',  # Overwrite output
                '-framerate', str(fps),
                '-i', str(temp_dir_path / 'frame_%06d.jpg'),
                '-c:v', 'libx264',
                '-preset', quality,
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-vf', f'scale={width}:{height}',
            ]
            
            if for_web:
                cmd.extend(['-movflags', '+faststart'])
            
            cmd.append(output_path)
            
            # Run FFmpeg
            print(f"🎬 Running FFmpeg command...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ FFmpeg error: {result.stderr}")
                return False
            
            # Verify the output file
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"✅ Video saved successfully: {output_path}")
                print(f"   Size: {file_size / 1024 / 1024:.2f} MB")
                return True
            else:
                print(f"❌ Output file not created: {output_path}")
                return False
                
    except Exception as e:
        print(f"❌ Error saving video with FFmpeg: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_fallback_opencv(frames: list, output_path: str, fps: int = 20) -> bool:
    """Fallback method using OpenCV if FFmpeg fails"""
    try:
        if not frames:
            return False
            
        height, width = frames[0].shape[:2]
        
        # Try different codecs
        codecs = [
            cv2.VideoWriter_fourcc(*'avc1'),
            cv2.VideoWriter_fourcc(*'h264'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            cv2.VideoWriter_fourcc(*'X264'),
        ]
        
        out = None
        for codec in codecs:
            try:
                out = cv2.VideoWriter(output_path, codec, fps, (width, height))
                if out.isOpened():
                    codec_str = chr(codec&0xFF) + chr((codec>>8)&0xFF) + chr((codec>>16)&0xFF) + chr((codec>>24)&0xFF)
                    print(f"✅ OpenCV using codec: {codec_str}")
                    break
            except:
                continue
        
        if not out or not out.isOpened():
            print("⚠️ Could not open VideoWriter with any codec")
            return False
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        if os.path.exists(output_path):
            print(f"✅ OpenCV fallback saved: {output_path}")
            return True
        return False
        
    except Exception as e:
        print(f"❌ OpenCV fallback error: {e}")
        return False