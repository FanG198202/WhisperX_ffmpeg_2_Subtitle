"""
WhisperX 音頻轉精準對齊字幕工具 (Google Colab 版)
=====================================================

這是一個在 Google Colab 環境中運行的互動式工具，利用 WhisperX 實現高精度的語音識別與逐字對齊，
並自動生成多種格式的字幕文件（SRT、VTT、TSV、TXT）。該工具特別適用於需要精確時間戳的場合，
如字幕製作、語言學習、視頻後期等。

功能特點
--------
- **精準人聲檢測**：基於 FFmpeg 的靜音檢測技術，自動提取音頻中的人聲片段（毫秒級）。
- **逐字對齊字幕**：使用 WhisperX 進行語音識別，並對每個字/詞進行精確的時間對齊。
- **多格式輸出**：自動生成 SRT、VTT、TSV、TXT 四種常見字幕格式，滿足不同平臺需求。
- **提示詞記憶**：支援常用提示詞（Initial Prompt）的保存與載入，提高特定場景的識別準確率。
- **靈活的文件來源**：可直接讀取 Google Drive 中的音頻文件，或手動上傳本地文件。
- **多語言支援**：支援繁體中文、簡體中文、美式英語、英式英語、日語、韓語、法語、德語、西班牙語等，並自動使用對應的檔案後綴。
- **模型選擇**：可根據 GPU 記憶體選擇 Whisper 模型大小（tiny/base/small/medium/large）。

依賴環境
--------
- Python 3.8+
- FFmpeg
- whisperx
- torch
- soundfile
- numpy
- ipywidgets (Colab 交互)
- google.colab (特定環境)

使用方法概述
------------
1. 在 Google Colab 中執行此腳本，它會自動安裝所需依賴。
2. 選擇音頻文件來源（Google Drive 或手動上傳）。
3. 設定字幕語言、初始提示詞和 Whisper 模型大小。
4. 點擊「開始生成字幕」按鈕，工具將自動完成人聲提取、語音識別、逐字對齊與字幕輸出。
5. 輸出文件會打包為 ZIP 壓縮包，可下載到本地或保存至 Google Drive。

函數簡介
--------
- `load_saved_prompts()`: 從本地 JSON 文件載入已保存的提示詞列表。
- `save_prompt(prompt)`: 將新的提示詞添加到保存列表並持久化。
- `extract_voice_segments(audio_path, output_json, min_volume, min_duration)`:
  使用 FFmpeg 分析音頻，提取人聲時間段（毫秒級），結果保存為 JSON。
- `format_time_srt(seconds)`: 將秒數轉換為 SRT 字幕的時間格式 (HH:MM:SS,mmm)。
- `format_time_vtt(seconds)`: 將秒數轉換為 VTT 字幕的時間格式 (HH:MM:SS.mmm)。
- `save_subtitle_formats(all_subtitles, base_filename, lang_suffix)`:
  根據對齊結果生成 SRT、VTT、TSV、TXT 四種格式的字幕文件，返回文件路徑列表。
- `transcribe_with_whisperx(audio_path, voice_segments, base_filename, lang, initial_prompt, model_size)`:
  對每個預先提取的人聲片段進行 WhisperX 識別與對齊，返回所有字幕片段的詳細資料。
- `main_interface()`: 構建 Colab 交互式界面，處理用戶輸入並協調整個處理流程。
- `install_dependencies()`: 安裝 FFmpeg 及必要的 Python 套件。

版本歷史
--------
- **v0.1.0.1** (2025-02-22): 初始版本，實現基本功能。
- **v0.1.0.2** (2025-02-23): 新增語言地區選擇，字幕檔案後綴自動對應（如 zh-TW, zh-CN, en-US 等）。
"""

import os
import json
import subprocess
import whisperx
import torch
import shutil
import zipfile
from pathlib import Path
from google.colab import drive, files
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# ===================== 初始化配置 =====================
# 加載已保存的prompt（如果存在）
PROMPT_SAVE_PATH = "saved_prompts.json"
def load_saved_prompts():
    """加載保存的提示詞列表"""
    if os.path.exists(PROMPT_SAVE_PATH):
        with open(PROMPT_SAVE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["請說普通話", "Speak clearly", "はっきり話してください"]

def save_prompt(prompt):
    """保存新的提示詞到列表"""
    prompts = load_saved_prompts()
    if prompt not in prompts:
        prompts.append(prompt)
        with open(PROMPT_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
    return prompts

# ===================== 核心功能函數 =====================
def extract_voice_segments(audio_path, output_json="voice_segments.json", min_volume=-30, min_duration=0.5):
    """使用FFmpeg提取音頻中的人聲時間段（毫秒級）"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音頻文件不存在: {audio_path}")

    # FFmpeg命令：分析音頻音量，輸出靜音/非靜音時間段
    cmd = [
        "ffmpeg",
        "-i", audio_path,
        "-af", f"silencedetect=noise={min_volume}dB:d={min_duration}",
        "-f", "null",
        "-"
    ]

    # 執行FFmpeg命令並捕獲輸出
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    output = result.stderr

    # 解析FFmpeg輸出，提取人聲時間段
    voice_segments = []
    silence_start = None

    for line in output.split("\n"):
        # 檢測靜音開始（意味着人聲結束）
        if "silence_start:" in line:
            try:
                # 提取靜音開始時間（秒）
                start_time = float(line.split("silence_start: ")[1].strip())
                if silence_start is not None:
                    # 計算人聲片段：上一個靜音結束 到 當前靜音開始
                    voice_start = int(silence_start * 1000)  # 轉毫秒
                    voice_end = int(start_time * 1000)
                    # 過濾過短的片段
                    if (voice_end - voice_start) > (min_duration * 1000):
                        voice_segments.append({"start": voice_start, "end": voice_end})
            except:
                continue

        # 檢測靜音結束（意味着人聲開始）
        elif "silence_end:" in line:
            try:
                # 提取靜音結束時間（秒）
                silence_start = float(line.split("silence_end: ")[1].split(" |")[0].strip())
            except:
                continue

    # 處理音頻末尾的人聲片段（如果最後不是靜音結束）
    duration_cmd = [
        "ffmpeg",
        "-i", audio_path,
        "-f", "null",
        "-"
    ]
    duration_result = subprocess.run(duration_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    for line in duration_result.stderr.split("\n"):
        if "Duration:" in line:
            try:
                duration_str = line.split("Duration: ")[1].split(",")[0].strip()
                h, m, s = duration_str.split(":")
                total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
                total_ms = int(total_seconds * 1000)
                # 如果最後有未結束的人聲片段
                if silence_start is not None:
                    voice_start = int(silence_start * 1000)
                    voice_end = total_ms
                    if (voice_end - voice_start) > (min_duration * 1000):
                        voice_segments.append({"start": voice_start, "end": voice_end})
            except:
                continue

    # 保存人聲片段到JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(voice_segments, f, ensure_ascii=False, indent=2)

    print(f"✅ 提取到 {len(voice_segments)} 個人聲片段，已保存到 {output_json}")
    return voice_segments

def format_time_srt(seconds):
    """轉換時間爲SRT格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

def format_time_vtt(seconds):
    """轉換時間爲VTT格式 (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"

def save_subtitle_formats(all_subtitles, base_filename, lang_suffix):
    """保存多種格式的字幕文件"""
    # 生成帶語言後綴的基礎文件名
    file_base = f"{base_filename}.{lang_suffix}"

    # 1. SRT格式
    srt_file = f"{file_base}.srt"
    with open(srt_file, "w", encoding="utf-8") as f:
        for sub in all_subtitles:
            f.write(f"{sub['id']}\n")
            f.write(f"{sub['start_srt']} --> {sub['end_srt']}\n")
            f.write(f"{sub['text']}\n\n")

    # 2. VTT格式
    vtt_file = f"{file_base}.vtt"
    with open(vtt_file, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for sub in all_subtitles:
            f.write(f"{sub['start_vtt']} --> {sub['end_vtt']}\n")
            f.write(f"{sub['text']}\n\n")

    # 3. TSV格式
    tsv_file = f"{file_base}.tsv"
    with open(tsv_file, "w", encoding="utf-8") as f:
        f.write("ID\tStart(ms)\tEnd(ms)\tStart\tEnd\tText\n")
        for sub in all_subtitles:
            start_ms = int(sub['start_sec'] * 1000)
            end_ms = int(sub['end_sec'] * 1000)
            f.write(f"{sub['id']}\t{start_ms}\t{end_ms}\t{sub['start_srt']}\t{sub['end_srt']}\t{sub['text']}\n")

    # 4. TXT格式（純文本）
    txt_file = f"{file_base}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        full_text = "".join([sub['text'] for sub in all_subtitles])
        f.write(full_text)

    return [srt_file, vtt_file, tsv_file, txt_file]

def transcribe_with_whisperx(audio_path, voice_segments, base_filename, lang="zh", initial_prompt="", model_size="base"):
    """使用WhisperX對人聲片段進行逐字精準對齊，生成多格式字幕"""
    # 設置設備（自動檢測GPU/CPU）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16 if torch.cuda.is_available() else 4
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    print(f"⚙️ 使用設備: {device}")
    print(f"⚙️ 模型大小: {model_size}, 語言: {lang}")
    print(f"⚙️ 初始提示詞: {initial_prompt}")

    # 1. 加載WhisperX模型
    model = whisperx.load_model(
        model_size,
        device,
        compute_type=compute_type,
        language=lang
    )

    # 2. 加載對齊模型
    model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)

    # 存儲所有字幕片段
    all_subtitles = []
    segment_id = 1

    # 3. 逐個處理人聲片段
    for seg in voice_segments:
        start_ms = seg["start"]
        end_ms = seg["end"]
        start_sec = start_ms / 1000
        end_sec = end_ms / 1000
        duration_sec = end_sec - start_sec

        print(f"\n🔤 處理片段 {segment_id}: {start_ms}ms - {end_ms}ms (時長: {duration_sec:.2f}秒)")

        # 臨時切割音頻片段
        temp_audio = f"temp_segment_{segment_id}.wav"
        try:
            # 使用FFmpeg切割音頻片段
            subprocess.run([
                "ffmpeg",
                "-ss", str(start_sec),
                "-to", str(end_sec),
                "-i", audio_path,
                "-vn", "-acodec", "pcm_s16le",
                "-y", temp_audio
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # 4. 識別音頻片段（帶初始提示詞）
            audio = whisperx.load_audio(temp_audio)
            result = model.transcribe(
                audio,
                batch_size=batch_size,
                initial_prompt=initial_prompt
            )

            # 5. 精準對齊（逐字級別）
            result_aligned = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=True
            )

            # 6. 處理對齊結果
            for word_seg in result_aligned["segments"]:
                for char in word_seg["char_alignments"]:
                    # 計算字符的絕對時間（加上片段起始時間）
                    char_start = start_sec + char["start"]
                    char_end = start_sec + char["end"]
                    char_text = char["char"]

                    # 過濾空字符
                    if char_text.strip() == "":
                        continue

                    # 轉換多種時間格式
                    start_srt = format_time_srt(char_start)
                    end_srt = format_time_srt(char_end)
                    start_vtt = format_time_vtt(char_start)
                    end_vtt = format_time_vtt(char_end)

                    # 添加到字幕列表
                    all_subtitles.append({
                        "id": len(all_subtitles) + 1,
                        "start_sec": char_start,
                        "end_sec": char_end,
                        "start_srt": start_srt,
                        "end_srt": end_srt,
                        "start_vtt": start_vtt,
                        "end_vtt": end_vtt,
                        "text": char_text
                    })

        except Exception as e:
            print(f"❌ 處理片段 {segment_id} 出錯: {e}")
        finally:
            # 刪除臨時音頻文件
            if os.path.exists(temp_audio):
                os.remove(temp_audio)

        segment_id += 1

    return all_subtitles

# ===================== Colab交互界面 =====================
def main_interface():
    clear_output()
    print("📌 音頻轉精準對齊字幕工具 (Google Colab 版)")
    print("="*50)

    # 1. 掛載Google Drive
    drive_mounted = False
    drive_path = "/content/drive/MyDrive/Conv2Sub"
    output_drive_path = f"{drive_path}/subtitle_output"

    try:
        drive.mount('/content/drive')
        drive_mounted = True
        print("✅ 已成功掛載Google雲端硬盤")

        # 創建目錄
        os.makedirs(drive_path, exist_ok=True)
        os.makedirs(output_drive_path, exist_ok=True)
        print(f"📂 雲端硬盤工作目錄: {drive_path}")

        # 列出Conv2Sub目錄下的音頻文件
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma']
        drive_audio_files = []
        for file in os.listdir(drive_path):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                drive_audio_files.append(file)

    except Exception as e:
        print(f"⚠️ 掛載雲端硬盤失敗: {e}")
        drive_audio_files = []

    # 2. 音頻文件選擇
    print("\n📁 音頻文件來源選擇:")
    if drive_audio_files:
        audio_source_options = ["📂 雲端硬盤文件"] + drive_audio_files + ["📤 手動上傳文件"]
    else:
        audio_source_options = ["📂 雲端硬盤文件（未檢測到）", "📤 手動上傳文件"]

    audio_source = widgets.Dropdown(
        options=audio_source_options,
        value=audio_source_options[0],
        description='選擇:',
        style={'description_width': 'initial'}
    )
    display(audio_source)

    # 3. 語言選擇（擴充地區後綴）
    print("\n🌐 字幕語言選擇:")
    # 語言映射：顯示名稱 -> (whisper語言代碼, 檔案後綴)
    lang_mapping = {
        "繁體中文": {"code": "zh", "suffix": "zh-TW"},
        "簡體中文": {"code": "zh", "suffix": "zh-CN"},
        "英文 (美國)": {"code": "en", "suffix": "en-US"},
        "英文 (英國)": {"code": "en", "suffix": "en-GB"},
        "日文": {"code": "ja", "suffix": "ja"},
        "韓文": {"code": "ko", "suffix": "ko"},
        "法語": {"code": "fr", "suffix": "fr"},
        "德語": {"code": "de", "suffix": "de"},
        "西班牙語": {"code": "es", "suffix": "es"},
        # 可根據需求繼續添加
    }
    lang_selector = widgets.Dropdown(
        options=list(lang_mapping.keys()),
        value="繁體中文",
        description='語言:',
        style={'description_width': 'initial'}
    )
    display(lang_selector)

    # 4. Initial Prompt選擇
    print("\n💡 初始提示詞 (Initial Prompt):")
    saved_prompts = load_saved_prompts()
    prompt_options = saved_prompts + ["[新增] 自定義提示詞"]
    prompt_selector = widgets.Dropdown(
        options=prompt_options,
        value=prompt_options[0],
        description='提示詞:',
        style={'description_width': 'initial'}
    )
    custom_prompt = widgets.Text(
        value="",
        placeholder="輸入自定義提示詞...",
        description='自定義:',
        style={'description_width': 'initial'},
        disabled=True
    )

    # 提示詞選擇聯動
    def on_prompt_change(change):
        if change['new'] == "[新增] 自定義提示詞":
            custom_prompt.disabled = False
        else:
            custom_prompt.disabled = True
            custom_prompt.value = ""

    prompt_selector.observe(on_prompt_change, names='value')
    display(prompt_selector)
    display(custom_prompt)

    # 5. 模型大小選擇 (整合 WhisperX 最新支援模型)
    print("\n⚙️ 模型大小選擇 (建議根據您的 GPU 顯存選擇):")
    model_options = {
        "Tiny (最快，精度低，需 ~1GB VRAM)": "tiny",
        "Base (平衡，需 ~1GB VRAM)": "base",
        "Small (較精準，需 ~2GB VRAM)": "small",
        "Medium (高精準，需 ~5GB VRAM)": "medium",
        "Large-v3-Turbo (極速推薦！精度接近 Large 但快 6 倍，需 ~6GB VRAM)": "large-v3-turbo",
        "Large-v2 (傳統最精準標竿，需 ~10GB VRAM)": "large-v2",
        "Large-v3 (目前多語系最強，需 ~10GB VRAM)": "large-v3"
    }

    model_selector = widgets.Dropdown(
        options=list(model_options.keys()),
        value="Large-v3-Turbo (極速推薦！精度接近 Large 但快 6 倍，需 ~6GB VRAM)", # 預設推薦 Turbo 版
        description='選擇模型:',
        style={'description_width': 'initial'},
        layout={'width': 'max-content'} # 自動調整寬度以顯示完整文字
    )

    display(model_selector)

    # 6. 執行按鈕
    print("\n🚀 開始轉換:")
    run_button = widgets.Button(
        description="開始生成字幕",
        button_style='success',
        icon='play'
    )
    display(run_button)

    # ===================== 執行邏輯 =====================
    def on_run_click(b):
        clear_output(wait=True)
        print("🚀 開始處理...")

        # 獲取選擇的參數
        selected_audio = audio_source.value
        selected_display = lang_selector.value
        lang_info = lang_mapping[selected_display]
        selected_lang_code = lang_info["code"]   # 傳給 WhisperX 的語言參數（基礎代碼）
        lang_suffix = lang_info["suffix"]        # 用於檔案名稱的後綴
        selected_model = model_options[model_selector.value]

        # 處理提示詞
        if prompt_selector.value == "[新增] 自定義提示詞" and custom_prompt.value.strip():
            selected_prompt = custom_prompt.value.strip()
            save_prompt(selected_prompt)
            print(f"💡 使用自定義提示詞並保存: {selected_prompt}")
        else:
            selected_prompt = prompt_selector.value

        # 處理音頻文件
        audio_file_path = ""
        is_drive_file = False

        if selected_audio.startswith("📤"):
            # 手動上傳文件
            print("\n📤 請上傳音頻文件...")
            uploaded = files.upload()
            if uploaded:
                audio_filename = list(uploaded.keys())[0]
                audio_file_path = f"/content/{audio_filename}"
                temp_dir = f"/content/temp_subtitles"
                os.makedirs(temp_dir, exist_ok=True)
                print(f"✅ 已上傳文件: {audio_filename}")
        elif selected_audio in drive_audio_files:
            # 雲端硬盤文件
            audio_file_path = f"{drive_path}/{selected_audio}"
            is_drive_file = True
            print(f"✅ 選擇雲端硬盤文件: {audio_file_path}")
        else:
            print("❌ 未選擇有效的音頻文件！")
            return

        # 提取文件名（不含擴展名）
        audio_basename = os.path.splitext(os.path.basename(audio_file_path))[0]

        try:
            # 步驟1：提取人聲片段
            print("\n🔍 正在分析音頻，提取人聲片段...")
            segments_json = f"{audio_basename}.voice_segments.{lang_suffix}.json"
            voice_segments = extract_voice_segments(audio_file_path, segments_json)

            if not voice_segments:
                print("❌ 未檢測到人聲片段！")
                return

            # 步驟2：WhisperX逐字對齊
            print("\n🎙️ 正在進行語音識別和逐字對齊...")
            all_subtitles = transcribe_with_whisperx(
                audio_file_path,
                voice_segments,
                audio_basename,
                lang=selected_lang_code,          # 傳遞基礎語言代碼
                initial_prompt=selected_prompt,
                model_size=selected_model
            )

            # 步驟3：保存多種格式字幕（使用地區後綴）
            print("\n💾 正在保存字幕文件...")
            subtitle_files = save_subtitle_formats(all_subtitles, audio_basename, lang_suffix)
            subtitle_files.append(segments_json)  # 加入人聲片段JSON

            # 步驟4：處理輸出文件
            if is_drive_file:
                # 雲端硬盤文件：打包保存到subtitle_output
                zip_filename = f"{audio_basename}.subtitles.{lang_suffix}.zip"
                zip_path = f"{output_drive_path}/{zip_filename}"

                # 打包文件
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in subtitle_files:
                        if os.path.exists(file):
                            zipf.write(file, os.path.basename(file))

                # 清理臨時文件
                for file in subtitle_files:
                    if os.path.exists(file):
                        os.remove(file)

                print(f"\n🎉 處理完成！")
                print(f"📦 字幕文件已打包保存到雲端硬盤:")
                print(f"   {zip_path}")
                print(f"\n💡 你可以在Google雲端硬盤的 Conv2Sub/subtitle_output 目錄找到該文件")

            else:
                # 手動上傳文件：保存到臨時目錄並提示下載
                for file in subtitle_files:
                    if os.path.exists(file):
                        shutil.move(file, temp_dir)

                print(f"\n🎉 處理完成！")
                print(f"📂 字幕文件已保存到臨時目錄: {temp_dir}")
                print(f"\n⚠️  重要提示：")
                print(f"   - Colab臨時文件會在會話結束後刪除")
                print(f"   - 請儘快下載以下文件到本地：")

                # 列出所有文件並提供下載鏈接
                for file in os.listdir(temp_dir):
                    file_path = f"{temp_dir}/{file}"
                    print(f"     - {file}")
                    display(HTML(f'<a href="files/{file_path}" download="{file}">📥 下載 {file}</a>'))

                # 提供打包下載
                zip_filename = f"{audio_basename}.subtitles.{lang_suffix}.zip"
                zip_path = f"/content/{zip_filename}"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in os.listdir(temp_dir):
                        file_path = f"{temp_dir}/{file}"
                        zipf.write(file_path, file)

                print(f"\n📦 也可以下載打包文件：")
                display(HTML(f'<a href="files/{zip_path}" download="{zip_filename}">📥 下載全部文件 ({zip_filename})</a>'))

        except Exception as e:
            print(f"\n❌ 處理過程出錯: {str(e)}")
            import traceback
            traceback.print_exc()

    run_button.on_click(on_run_click)

# ===================== 安裝依賴 =====================
def install_dependencies():
    print("📦 正在安裝必要依賴...")
    # 安裝FFmpeg
    subprocess.run(["apt", "update"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["apt", "install", "-y", "ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 安裝Python包 - 使用 subprocess.run 而不是 !pip
    subprocess.run(["pip", "install", "-q", "ffmpeg-python", "whisperx", "torch", "soundfile", "numpy"])

    print("✅ 依賴安裝完成！")

# 執行安裝和啓動界面
if __name__ == "__main__":
    install_dependencies()
    main_interface()