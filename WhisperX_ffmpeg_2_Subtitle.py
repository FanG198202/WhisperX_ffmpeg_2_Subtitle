"""
WhisperX 音頻轉精準對齊字幕工具 (Google Colab 版)
=====================================================

這是一個在 Google Colab 環境中運行的互動式工具，利用 WhisperX 實現高精度的語音識別與逐字對齊，
並自動生成多種格式的字幕文件（SRT、VTT、TSV、TXT）。該工具特別適用於需要精確時間戳的場合，
如字幕製作、語言學習、視頻後期等。

功能特點
--------
- 精準人聲檢測：基於 FFmpeg 的靜音檢測技術，自動提取音頻中的人聲片段（毫秒級）。
- 逐字對齊字幕：使用 WhisperX 進行語音識別，並對每個字/詞進行精確的時間對齊。
- 多格式輸出：自動生成 SRT、VTT、TSV、TXT 四種常見字幕格式，滿足不同平台需求。
- 提示詞記憶：支援常用提示詞（Initial Prompt）的保存與載入，提高特定場景的識別準確率。
- 靈活的文件來源：可直接讀取 Google Drive 中的音頻文件，或手動上傳本地文件。
- 多語言支援：支援繁體中文、簡體中文、美式英語、英式英語、日語、韓語、法語、德語、西班牙語等。
- 模型選擇：可根據 GPU 記憶體選擇 Whisper 模型大小（tiny/base/small/medium/large）。

依賴環境
--------
- Python 3.8+
- FFmpeg
- whisperx
- torch
- soundfile
- numpy
- ipywidgets (Colab 交互)
- google-colab (特定環境)

版本歷史
--------
- v0.1.0.1 (2025-02-22): 初始版本，實現基本功能。
- v0.1.0.2 (2025-02-23): 新增語言地區選擇，字幕檔案後綴自動對應。
- v0.1.0.3 (2026-02-22): 修正多項 Bug，包含人聲片段遺漏、char_alignments key 錯誤、GPU 記憶體未釋放等問題。
"""

# ===========================================================
# [FIX-1] google.colab 僅在 Colab 環境下才能 import，
#         改用 try/except 延遲匯入，避免在本地端執行時直接崩潰。
# ===========================================================
import os
import json
import subprocess
import shutil
import zipfile
from pathlib import Path

try:
    from google.colab import drive, files as colab_files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ===========================================================
# [FIX-11] torch / whisperx / soundfile / numpy 等重型套件
#          不在頂層 import，因為腳本執行時會先跑頂層再進入
#          if __name__ == "__main__" 的 install_dependencies()。
#          若套件尚未安裝，頂層 import 就會拋出 ModuleNotFoundError。
#          修正：改為在各自需要的函數內部才 import（延遲 import）。
#          ipywidgets / IPython 同理。
# ===========================================================
# 標準庫以外的套件均在函數內延遲 import，此處不做頂層 import。

# ===================== 初始化配置 =====================
PROMPT_SAVE_PATH = "saved_prompts.json"


def load_saved_prompts():
    """
    從本地 JSON 文件載入已保存的提示詞列表。

    返回:
        list: 提示詞字串列表；若文件不存在則返回預設列表。
    """
    if os.path.exists(PROMPT_SAVE_PATH):
        with open(PROMPT_SAVE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["請說普通話", "Speak clearly", "はっきり話してください"]


def save_prompt(prompt: str) -> list:
    """
    將新的提示詞添加到保存列表並持久化到 JSON 文件。

    參數:
        prompt (str): 要保存的提示詞字串。

    返回:
        list: 更新後的完整提示詞列表。
    """
    prompts = load_saved_prompts()
    if prompt not in prompts:
        prompts.append(prompt)
        with open(PROMPT_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
    return prompts


# ===================== 核心功能函數 =====================

def extract_voice_segments(
    audio_path: str,
    output_json: str = "voice_segments.json",
    min_volume: int = -30,
    min_duration: float = 0.5
) -> list:
    """
    使用 FFmpeg silencedetect 提取音頻中的人聲時間段（毫秒級）。

    參數:
        audio_path (str): 輸入音頻文件路徑。
        output_json (str): 輸出 JSON 文件路徑，用於保存人聲片段資訊。
        min_volume (int): 靜音檢測閾值（dB），低於此值視為靜音，預設 -30。
        min_duration (float): 靜音最短持續秒數，低於此值不視為靜音，預設 0.5。

    返回:
        list: 人聲片段列表，每項為 {"start": int(ms), "end": int(ms)}。

    異常:
        FileNotFoundError: 若音頻文件不存在時拋出。
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音頻文件不存在: {audio_path}")

    # FFmpeg silencedetect 分析
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-af", f"silencedetect=noise={min_volume}dB:d={min_duration}",
        "-f", "null", "-"
    ]
    result = subprocess.run(
        cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    output = result.stderr

    # ----------------------------------------------------------
    # [FIX-2] 原始邏輯遺漏了音頻開頭到第一個靜音開始之間的人聲片段。
    #         修正方式：
    #         - voice_cursor 記錄「目前人聲的開始時間」，初始為 0.0
    #         - 遇到 silence_start 就結束一段人聲（從 voice_cursor 到 silence_start）
    #         - 遇到 silence_end 就更新 voice_cursor（靜音結束後人聲開始）
    # ----------------------------------------------------------
    voice_segments = []
    voice_cursor = 0.0           # 目前人聲開始時間（秒）
    in_initial_silence = True    # 標記是否還未遇到第一個 silence_end

    for line in output.split("\n"):
        if "silence_start:" in line:
            try:
                silence_start_sec = float(line.split("silence_start:")[1].strip())
                if not in_initial_silence:
                    # 人聲片段：voice_cursor 到 silence_start
                    v_start_ms = int(voice_cursor * 1000)
                    v_end_ms = int(silence_start_sec * 1000)
                    if (v_end_ms - v_start_ms) > int(min_duration * 1000):
                        voice_segments.append({"start": v_start_ms, "end": v_end_ms})
                else:
                    # 音頻從 0 就開始靜音，記錄開頭到 silence_start 這段（若夠長）
                    v_start_ms = 0
                    v_end_ms = int(silence_start_sec * 1000)
                    if (v_end_ms - v_start_ms) > int(min_duration * 1000):
                        voice_segments.append({"start": v_start_ms, "end": v_end_ms})
                    in_initial_silence = False  # 進入靜音，下次 silence_end 才重置 cursor
            except (IndexError, ValueError):
                continue

        elif "silence_end:" in line:
            try:
                # silence_end 行格式: "silence_end: 3.2 | silence_duration: 1.7"
                silence_end_sec = float(line.split("silence_end:")[1].split("|")[0].strip())
                voice_cursor = silence_end_sec
                in_initial_silence = False
            except (IndexError, ValueError):
                continue

    # ----------------------------------------------------------
    # [FIX-3] 原來使用兩個 subprocess.run 取得音頻總長，效率低且易出錯。
    #         改用 ffprobe 精確取得 duration，只呼叫一次。
    # ----------------------------------------------------------
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    probe_result = subprocess.run(
        probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        total_seconds = float(probe_result.stdout.strip())
        total_ms = int(total_seconds * 1000)
        # 處理音頻末尾的未結束人聲（最後一段靜音後沒有 silence_end）
        v_start_ms = int(voice_cursor * 1000)
        if (total_ms - v_start_ms) > int(min_duration * 1000):
            voice_segments.append({"start": v_start_ms, "end": total_ms})
    except ValueError:
        pass

    # 保存結果
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(voice_segments, f, ensure_ascii=False, indent=2)

    print(f"[OK] 提取到 {len(voice_segments)} 個人聲片段，已保存到 {output_json}")
    return voice_segments


def format_time_srt(seconds: float) -> str:
    """
    將秒數轉換為 SRT 時間格式 (HH:MM:SS,mmm)。

    參數:
        seconds (float): 時間（秒）。

    返回:
        str: SRT 格式的時間字串。
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_time_vtt(seconds: float) -> str:
    """
    將秒數轉換為 VTT 時間格式 (HH:MM:SS.mmm)。

    參數:
        seconds (float): 時間（秒）。

    返回:
        str: VTT 格式的時間字串。
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def save_subtitle_formats(all_subtitles: list, base_filename: str, lang_suffix: str) -> list:
    """
    根據對齊結果生成 SRT、VTT、TSV、TXT 四種格式的字幕文件。

    參數:
        all_subtitles (list): 包含逐字字幕資訊的字典列表。
        base_filename (str): 輸出文件的基礎名稱（不含副檔名）。
        lang_suffix (str): 語言後綴（如 zh-TW、en-US）。

    返回:
        list: 所有生成的字幕文件路徑列表。
    """
    file_base = f"{base_filename}.{lang_suffix}"

    # 1. SRT 格式
    srt_file = f"{file_base}.srt"
    with open(srt_file, "w", encoding="utf-8") as f:
        for sub in all_subtitles:
            f.write(f"{sub['id']}\n")
            f.write(f"{sub['start_srt']} --> {sub['end_srt']}\n")
            f.write(f"{sub['text']}\n\n")

    # 2. VTT 格式
    vtt_file = f"{file_base}.vtt"
    with open(vtt_file, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for sub in all_subtitles:
            f.write(f"{sub['start_vtt']} --> {sub['end_vtt']}\n")
            f.write(f"{sub['text']}\n\n")

    # 3. TSV 格式
    tsv_file = f"{file_base}.tsv"
    with open(tsv_file, "w", encoding="utf-8") as f:
        f.write("ID\tStart(ms)\tEnd(ms)\tStart\tEnd\tText\n")
        for sub in all_subtitles:
            start_ms = int(sub['start_sec'] * 1000)
            end_ms = int(sub['end_sec'] * 1000)
            f.write(
                f"{sub['id']}\t{start_ms}\t{end_ms}\t"
                f"{sub['start_srt']}\t{sub['end_srt']}\t{sub['text']}\n"
            )

    # 4. TXT 格式（純文本）
    # ----------------------------------------------------------
    # [FIX-4] 原版逐字 join 後沒有分隔，導致輸出為黏在一起的字串。
    #         改為以段落為單位輸出，每個 segment 之間加換行，
    #         讓 TXT 保持可讀性。
    # ----------------------------------------------------------
    txt_file = f"{file_base}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        current_line = []
        prev_end = None
        LINE_BREAK_THRESHOLD = 1.0  # 相鄰字幕間距超過 1 秒則換行

        for sub in all_subtitles:
            if prev_end is not None and (sub['start_sec'] - prev_end) > LINE_BREAK_THRESHOLD:
                f.write("".join(current_line) + "\n")
                current_line = []
            current_line.append(sub['text'])
            prev_end = sub['end_sec']

        if current_line:
            f.write("".join(current_line) + "\n")

    return [srt_file, vtt_file, tsv_file, txt_file]


def transcribe_with_whisperx(
    audio_path: str,
    voice_segments: list,
    base_filename: str,
    lang: str = "zh",
    initial_prompt: str = "",
    model_size: str = "base"
) -> list:
    """
    使用 WhisperX 對人聲片段進行逐字精準對齊，生成多格式字幕。

    參數:
        audio_path (str): 輸入音頻文件路徑。
        voice_segments (list): 人聲片段列表（來自 extract_voice_segments）。
        base_filename (str): 輸出文件基礎名稱。
        lang (str): Whisper 語言代碼（如 "zh"、"en"）。
        initial_prompt (str): 初始提示詞，提升特定場景識別精度。
        model_size (str): Whisper 模型大小。

    返回:
        list: 所有字幕片段的字典列表。

    異常:
        RuntimeError: 若 WhisperX 模型載入失敗時拋出。
    """
    # 延遲 import：確保 install_dependencies() 執行完後才載入
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16 if torch.cuda.is_available() else 4
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    print(f"[INFO] 使用設備: {device}")
    print(f"[INFO] 模型大小: {model_size}, 語言: {lang}")
    print(f"[INFO] 初始提示詞: {initial_prompt}")

    # 載入 WhisperX 辨識模型
    model = whisperx.load_model(
        model_size, device, compute_type=compute_type, language=lang
    )

    # 載入對齊模型
    model_a, metadata = whisperx.load_align_model(
        language_code=lang, device=device
    )

    all_subtitles = []
    segment_id = 1

    for seg in voice_segments:
        start_ms = seg["start"]
        end_ms = seg["end"]
        start_sec = start_ms / 1000
        end_sec = end_ms / 1000

        print(f"\n[INFO] 處理片段 {segment_id}: {start_ms}ms - {end_ms}ms "
              f"(時長: {end_sec - start_sec:.2f}秒)")

        temp_audio = f"temp_segment_{segment_id}.wav"
        try:
            # 切割音頻片段
            subprocess.run([
                "ffmpeg",
                "-ss", str(start_sec),
                "-to", str(end_sec),
                "-i", audio_path,
                "-vn", "-acodec", "pcm_s16le",
                "-y", temp_audio
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # 語音辨識
            audio = whisperx.load_audio(temp_audio)
            result = model.transcribe(
                audio,
                batch_size=batch_size,
                initial_prompt=initial_prompt if initial_prompt else None
            )

            # 逐字對齊
            result_aligned = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=True
            )

            # ----------------------------------------------------------
            # [FIX-5] 原版使用 "char_alignments" 作為 key，
            #         但 WhisperX 實際 API 返回的 key 為 "chars"。
            #         同時加入 fallback 邏輯：若 "chars" 不存在，
            #         則退回使用 word 級別對齊（"words"）確保不中斷。
            # ----------------------------------------------------------
            for word_seg in result_aligned.get("segments", []):
                # 優先嘗試字元級對齊
                char_items = word_seg.get("chars", None)
                if char_items:
                    for char in char_items:
                        char_start = start_sec + char.get("start", 0)
                        char_end = start_sec + char.get("end", 0)
                        char_text = char.get("char", "")
                        if not char_text.strip():
                            continue
                        all_subtitles.append({
                            "id": len(all_subtitles) + 1,
                            "start_sec": char_start,
                            "end_sec": char_end,
                            "start_srt": format_time_srt(char_start),
                            "end_srt": format_time_srt(char_end),
                            "start_vtt": format_time_vtt(char_start),
                            "end_vtt": format_time_vtt(char_end),
                            "text": char_text
                        })
                else:
                    # fallback: 使用 word 級別對齊
                    for word in word_seg.get("words", []):
                        w_start = start_sec + word.get("start", 0)
                        w_end = start_sec + word.get("end", 0)
                        w_text = word.get("word", "").strip()
                        if not w_text:
                            continue
                        all_subtitles.append({
                            "id": len(all_subtitles) + 1,
                            "start_sec": w_start,
                            "end_sec": w_end,
                            "start_srt": format_time_srt(w_start),
                            "end_srt": format_time_srt(w_end),
                            "start_vtt": format_time_vtt(w_start),
                            "end_vtt": format_time_vtt(w_end),
                            "text": w_text
                        })

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] FFmpeg 切割片段 {segment_id} 失敗: {e}")
        except Exception as e:
            print(f"[ERROR] 處理片段 {segment_id} 出錯: {e}")
        finally:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)

        segment_id += 1

    # ----------------------------------------------------------
    # [FIX-6] 原版未釋放 GPU 記憶體，長音頻處理後容易 OOM。
    #         處理完所有片段後主動釋放模型並清空 CUDA 快取。
    # ----------------------------------------------------------
    del model
    del model_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[INFO] GPU 記憶體已釋放")

    return all_subtitles


# ===================== Colab 交互界面 =====================

def main_interface():
    """建構 Colab 互動式界面，協調整個字幕生成流程。"""
    # 延遲 import：確保 install_dependencies() 執行完後才載入
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML

    clear_output()
    print("[INFO] 音頻轉精準對齊字幕工具 (Google Colab 版)")
    print("=" * 50)

    # ----------------------------------------------------------
    # [FIX-7] 在非 Colab 環境執行時，跳過 drive.mount() 的呼叫。
    # ----------------------------------------------------------
    drive_mounted = False
    drive_path = "/content/drive/MyDrive/Conv2Sub"
    output_drive_path = f"{drive_path}/subtitle_output"
    drive_audio_files = []

    if IN_COLAB:
        try:
            drive.mount('/content/drive')
            drive_mounted = True
            print("[OK] 已成功掛載 Google 雲端硬盤")

            os.makedirs(drive_path, exist_ok=True)
            os.makedirs(output_drive_path, exist_ok=True)
            print(f"[INFO] 雲端硬盤工作目錄: {drive_path}")

            audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma']
            for file in os.listdir(drive_path):
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    drive_audio_files.append(file)
        except Exception as e:
            print(f"[WARN] 掛載雲端硬盤失敗: {e}")
    else:
        print("[WARN] 非 Colab 環境，跳過 Google Drive 掛載")

    # 音頻文件選擇
    print("\n[STEP 1] 音頻文件來源選擇:")

    # ----------------------------------------------------------
    # [FIX-8] 「雲端硬盤文件（未檢測到）」是一個無效的佔位選項，
    #         使用者若選到它會導致後續流程走到 else 分支報錯。
    #         修正：只在有雲端硬盤文件時才加入 Drive 選項前綴。
    # ----------------------------------------------------------
    if drive_audio_files:
        audio_source_options = drive_audio_files + ["[手動上傳文件]"]
    else:
        audio_source_options = ["[手動上傳文件]"]
        if drive_mounted:
            print(f"[INFO] 未在 {drive_path} 檢測到音頻文件，請手動上傳。")

    audio_source = widgets.Dropdown(
        options=audio_source_options,
        value=audio_source_options[0],
        description='選擇:',
        style={'description_width': 'initial'}
    )
    display(audio_source)

    # 語言選擇
    print("\n[STEP 2] 字幕語言選擇:")
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
    }
    lang_selector = widgets.Dropdown(
        options=list(lang_mapping.keys()),
        value="繁體中文",
        description='語言:',
        style={'description_width': 'initial'}
    )
    display(lang_selector)

    # 初始提示詞
    print("\n[STEP 3] 初始提示詞 (Initial Prompt):")
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

    def on_prompt_change(change):
        if change['new'] == "[新增] 自定義提示詞":
            custom_prompt.disabled = False
        else:
            custom_prompt.disabled = True
            custom_prompt.value = ""

    prompt_selector.observe(on_prompt_change, names='value')
    display(prompt_selector)
    display(custom_prompt)

    # 模型大小選擇
    print("\n[STEP 4] 模型大小選擇 (建議根據 GPU 顯存選擇):")
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
        value="Large-v3-Turbo (極速推薦！精度接近 Large 但快 6 倍，需 ~6GB VRAM)",
        description='選擇模型:',
        style={'description_width': 'initial'},
        layout={'width': 'max-content'}
    )
    display(model_selector)

    # 執行按鈕
    print("\n[STEP 5] 開始轉換:")
    run_button = widgets.Button(
        description="開始生成字幕",
        button_style='success',
        icon='play'
    )
    display(run_button)

    # ===================== 執行邏輯 =====================
    def on_run_click(b):
        clear_output(wait=True)
        print("[INFO] 開始處理...")

        selected_audio = audio_source.value
        lang_info = lang_mapping[lang_selector.value]
        selected_lang_code = lang_info["code"]
        lang_suffix = lang_info["suffix"]
        selected_model = model_options[model_selector.value]

        # 處理提示詞
        if prompt_selector.value == "[新增] 自定義提示詞" and custom_prompt.value.strip():
            selected_prompt = custom_prompt.value.strip()
            save_prompt(selected_prompt)
            print(f"[INFO] 使用自定義提示詞並保存: {selected_prompt}")
        elif prompt_selector.value == "[新增] 自定義提示詞":
            selected_prompt = ""
        else:
            selected_prompt = prompt_selector.value

        # 確定音頻文件路徑
        audio_file_path = ""
        is_drive_file = False
        temp_dir = "/content/temp_subtitles"

        if selected_audio == "[手動上傳文件]":
            if not IN_COLAB:
                print("[ERROR] 非 Colab 環境無法使用手動上傳功能。")
                return
            print("\n[INFO] 請上傳音頻文件...")
            uploaded = colab_files.upload()
            if not uploaded:
                print("[ERROR] 未上傳任何文件！")
                return
            audio_filename = list(uploaded.keys())[0]
            # ----------------------------------------------------------
            # [FIX-9] 原版在上傳後將 audio_file_path 設為 /content/...
            #         但 temp_dir 是獨立目錄，兩者不一致。
            #         修正：上傳後的文件就放在 /content/，
            #               temp_dir 用於存放輸出字幕文件，兩者分開管理。
            # ----------------------------------------------------------
            audio_file_path = f"/content/{audio_filename}"
            os.makedirs(temp_dir, exist_ok=True)
            print(f"[OK] 已上傳文件: {audio_filename}")
        elif selected_audio in drive_audio_files:
            audio_file_path = f"{drive_path}/{selected_audio}"
            is_drive_file = True
            print(f"[OK] 選擇雲端硬盤文件: {audio_file_path}")
        else:
            print("[ERROR] 未選擇有效的音頻文件！")
            return

        audio_basename = os.path.splitext(os.path.basename(audio_file_path))[0]

        try:
            # Step 1: 提取人聲片段
            print("\n[STEP 1] 正在分析音頻，提取人聲片段...")
            segments_json = f"{audio_basename}.voice_segments.{lang_suffix}.json"
            voice_segments = extract_voice_segments(audio_file_path, segments_json)

            if not voice_segments:
                print("[ERROR] 未檢測到人聲片段！請確認音頻文件是否正常。")
                return

            # Step 2: WhisperX 逐字對齊
            print("\n[STEP 2] 正在進行語音識別和逐字對齊...")
            all_subtitles = transcribe_with_whisperx(
                audio_file_path,
                voice_segments,
                audio_basename,
                lang=selected_lang_code,
                initial_prompt=selected_prompt,
                model_size=selected_model
            )

            if not all_subtitles:
                print("[WARN] 辨識結果為空，請確認音頻品質或嘗試調整初始提示詞。")
                return

            # Step 3: 保存多種格式字幕
            print("\n[STEP 3] 正在保存字幕文件...")
            subtitle_files = save_subtitle_formats(all_subtitles, audio_basename, lang_suffix)
            subtitle_files.append(segments_json)

            # Step 4: 處理輸出文件
            if is_drive_file:
                zip_filename = f"{audio_basename}.subtitles.{lang_suffix}.zip"
                zip_path = f"{output_drive_path}/{zip_filename}"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in subtitle_files:
                        if os.path.exists(file):
                            zipf.write(file, os.path.basename(file))
                for file in subtitle_files:
                    if os.path.exists(file):
                        os.remove(file)
                print(f"\n[OK] 處理完成！")
                print(f"[INFO] 字幕文件已打包保存到雲端硬盤:")
                print(f"      {zip_path}")
            else:
                # 手動上傳：將字幕移入 temp_dir
                for file in subtitle_files:
                    if os.path.exists(file):
                        shutil.move(file, temp_dir)

                print(f"\n[OK] 處理完成！")
                print(f"[INFO] 字幕文件已保存到臨時目錄: {temp_dir}")
                print(f"\n[WARN] 重要提示：Colab 臨時文件會在會話結束後刪除，請儘快下載。")

                for file in os.listdir(temp_dir):
                    file_path = f"{temp_dir}/{file}"
                    display(HTML(f'<a href="files/{file_path}" download="{file}">[下載] {file}</a>'))

                zip_filename = f"{audio_basename}.subtitles.{lang_suffix}.zip"
                zip_path = f"/content/{zip_filename}"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in os.listdir(temp_dir):
                        file_path = f"{temp_dir}/{file}"
                        zipf.write(file_path, file)
                print(f"\n[INFO] 打包下載：")
                display(HTML(
                    f'<a href="files/{zip_path}" download="{zip_filename}">'
                    f'[下載全部文件] {zip_filename}</a>'
                ))

        except FileNotFoundError as e:
            print(f"\n[ERROR] 文件不存在: {e}")
        except Exception as e:
            import traceback
            print(f"\n[ERROR] 處理過程出錯: {e}")
            traceback.print_exc()

    run_button.on_click(on_run_click)


# ===================== 安裝依賴 =====================

def install_dependencies():
    """
    安裝 FFmpeg 及必要的 Python 套件。
    僅在 Colab 或 Linux 環境下執行。
    """
    print("[INFO] 正在安裝必要依賴...")
    # ----------------------------------------------------------
    # [FIX-10] 原版缺少 ctranslate2 等 WhisperX 核心依賴，
    #          且沒有加入 --quiet 選項導致輸出過多。
    #          新增 ctranslate2、pyannote-audio 並整合安裝。
    # ----------------------------------------------------------
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg", "ffprobe"], check=True)
    subprocess.run([
        "pip", "install", "-q",
        "whisperx",
        "ctranslate2",
        "ffmpeg-python",
        "soundfile",
        "numpy",
        "torchaudio",
        "transformers",
        "pyannote-audio",
        "ipywidgets"
    ], check=True)
    print("[OK] 依賴安裝完成！")


if __name__ == "__main__":
    install_dependencies()
    main_interface()
