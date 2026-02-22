"""
基本功能測試
"""
import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# 導入主模塊
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from WhisperX_ffmpeg_2_Subtitle import (
    format_time_srt,
    format_time_vtt,
    load_saved_prompts,
    save_prompt
)

class TestSubtitleTools(unittest.TestCase):
    """測試字幕工具函數"""

    def test_format_time_srt(self):
        """測試 SRT 時間格式化"""
        # 測試基本轉換
        self.assertEqual(format_time_srt(0), "00:00:00,000")
        self.assertEqual(format_time_srt(1.5), "00:00:01,500")
        self.assertEqual(format_time_srt(3661.123), "01:01:01,123")

    def test_format_time_vtt(self):
        """測試 VTT 時間格式化"""
        self.assertEqual(format_time_vtt(0), "00:00:00.000")
        self.assertEqual(format_time_vtt(1.5), "00:00:01.500")
        self.assertEqual(format_time_vtt(3661.123), "01:01:01.123")

    def test_prompt_save_load(self):
        """測試提示詞保存和載入"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 模擬配置文件路徑
            with patch('WhisperX_ffmpeg_2_Subtitle.PROMPT_SAVE_PATH',
                      os.path.join(tmpdir, 'saved_prompts.json')):

                # 測試載入默認提示詞
                prompts = load_saved_prompts()
                self.assertIsInstance(prompts, list)

                # 測試保存新提示詞
                new_prompt = "測試提示詞"
                updated = save_prompt(new_prompt)
                self.assertIn(new_prompt, updated)

                # 測試載入保存的提示詞
                loaded = load_saved_prompts()
                self.assertIn(new_prompt, loaded)

class TestVoiceSegments(unittest.TestCase):
    """測試人聲片段提取（模擬）"""

    @patch('subprocess.run')
    def test_extract_voice_segments_mock(self, mock_run):
        """模擬 FFmpeg 輸出測試"""
        from WhisperX_ffmpeg_2_Subtitle import extract_voice_segments

        # 模擬 FFmpeg 輸出
        mock_result = MagicMock()
        mock_result.stderr = """
[silencedetect @ 0x55d6b8a3bcc0] silence_start: 1.5
[silencedetect @ 0x55d6b8a3bcc0] silence_end: 3.2 | silence_duration: 1.7
[silencedetect @ 0x55d6b8a3bcc0] silence_start: 5.8
        """
        mock_run.return_value = mock_result

        # 模擬持續時間查詢
        mock_duration = MagicMock()
        mock_duration.stderr = "Duration: 00:01:00.00, start: 0.000000, bitrate: 128 kb/s"

        def side_effect(*args, **kwargs):
            if "Duration:" in str(kwargs.get('stderr', '')) or \
               (args and "Duration:" in str(args)):
                return mock_duration
            return mock_result

        mock_run.side_effect = side_effect

        # 執行測試
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp_audio:
            result = extract_voice_segments(tmp_audio.name, "test.json")
            self.assertIsInstance(result, list)

if __name__ == '__main__':
    unittest.main()