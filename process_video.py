import os
import subprocess
import shutil

def convert_videos_to_mp3():
    # Check for videos directory (supports both 'videos' and legacy 'vedeos')
    video_dir = None
    for candidate in ["videos", "vedeos"]:
        if os.path.exists(candidate) and os.path.isdir(candidate):
            video_dir = candidate
            break

    if not video_dir:
        print("❌ Error: Neither 'videos' nor 'vedeos' directory was found.")
        return

    # Check ffmpeg installation
    if not shutil.which("ffmpeg"):
        print("❌ Error: ffmpeg is not installed or not in PATH.")
        return

    os.makedirs("audios", exist_ok=True)
    video_extensions = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv")
    files = [f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)]

    if not files:
        print(f"⚠️ No video files found in '{video_dir}'.")
        return

    print(f"🎬 Found {len(files)} video(s) in '{video_dir}'. Starting conversion to mp3...")

    for file in files:
        # Extract tutorial number or clean name
        if "#" in file:
            part = file.split("#")[1]
            tutorial_number = os.path.splitext(part)[0].strip()
        else:
            tutorial_number = os.path.splitext(file)[0].strip()

        input_path = os.path.join(video_dir, file)
        output_path = os.path.join("audios", f"{tutorial_number}.mp3")

        print(f"Converting: {file} -> {output_path}")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "libmp3lame", "-q:a", "2", output_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"  ✅ Saved: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to convert {file}: {e}")

    print("🎉 All video conversions completed!")

if __name__ == "__main__":
    convert_videos_to_mp3()
