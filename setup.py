from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="whisperx-ffmpeg-subtitle",
    version="0.1.0.2",
    author="WhisperX Subtitle Tool Contributors",
    author_email="your.email@example.com",
    description="WhisperX 音頻轉精準對齊字幕工具 - 使用 WhisperX 和 FFmpeg 實現高精度語音識別與逐字對齊",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/你的用戶名/WhisperX_ffmpeg_2_Subtitle",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Multimedia :: Video :: Conversion",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "whisperx-subtitle=WhisperX_ffmpeg_2_Subtitle:main",
        ],
    },
    include_package_data=True,
)