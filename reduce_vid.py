import subprocess
import os

def resize_video(input_path, output_path, target_width, target_height):
    """
    Resize a video to a target size with padding to maintain the aspect ratio.

    Args:
    - input_path (str): Path to the input video.
    - output_path (str): Path where the resized video will be saved.
    - target_width (int): Target width of the video.
    - target_height (int): Target height of the video.
    """
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', f"scale=w={target_width}:h={target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2",
        '-c:a', 'copy',  # Copy the audio stream without re-encoding
        output_path
    ]
    subprocess.run(cmd)

# Example usage:
input_videos_folder = 'assets/videos/'
output_videos_folder = 'assets/videos_reduced/'
target_width = 854/4
target_height = 480/4

if not os.path.exists(output_videos_folder):
    os.makedirs(output_videos_folder)

for video_name in os.listdir(input_videos_folder):
    input_path = os.path.join(input_videos_folder, video_name)
    output_path = os.path.join(output_videos_folder, video_name)
    resize_video(input_path, output_path, target_width, target_height)
    print(f'Resized {video_name} and saved to {output_path}')
