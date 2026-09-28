from __future__ import annotations

import unittest

from kura.backends.ai_toolkit import (
    AI_TOOLKIT_IMAGE_SUFFIXES,
    AI_TOOLKIT_VIDEO_SUFFIXES,
)
from kura.backends.musubi_datasets import MUSUBI_AUDIO_SUFFIXES, MUSUBI_IMAGE_SUFFIXES, MUSUBI_VIDEO_SUFFIXES
from kura.backends.sd_scripts_datasets import SD_SCRIPTS_IMAGE_SUFFIXES
from kura.media_types import (
    KNOWN_AUDIO_SUFFIXES,
    KNOWN_IMAGE_SUFFIXES,
    KNOWN_MEDIA_SUFFIXES,
    KNOWN_VIDEO_SUFFIXES,
    frozen_suffixes,
)


class MediaTypeRegistryTests(unittest.TestCase):
    def test_core_registry_is_a_broad_media_vocabulary(self) -> None:
        self.assertIn(".jxl", KNOWN_IMAGE_SUFFIXES)
        self.assertIn(".mpeg", KNOWN_VIDEO_SUFFIXES)
        self.assertIn(".wmv", KNOWN_VIDEO_SUFFIXES)
        self.assertIn(".m4a", KNOWN_AUDIO_SUFFIXES)
        self.assertEqual(
            KNOWN_MEDIA_SUFFIXES,
            KNOWN_IMAGE_SUFFIXES | KNOWN_VIDEO_SUFFIXES | KNOWN_AUDIO_SUFFIXES,
        )

    def test_pinned_backend_loader_capabilities_are_core_subsets(self) -> None:
        self.assertLessEqual(AI_TOOLKIT_IMAGE_SUFFIXES, KNOWN_IMAGE_SUFFIXES)
        self.assertLessEqual(AI_TOOLKIT_VIDEO_SUFFIXES, KNOWN_VIDEO_SUFFIXES)
        self.assertLessEqual(MUSUBI_IMAGE_SUFFIXES, KNOWN_IMAGE_SUFFIXES)
        self.assertLessEqual(MUSUBI_VIDEO_SUFFIXES, KNOWN_VIDEO_SUFFIXES)
        self.assertLessEqual(MUSUBI_AUDIO_SUFFIXES, KNOWN_AUDIO_SUFFIXES)
        self.assertLessEqual(SD_SCRIPTS_IMAGE_SUFFIXES, KNOWN_IMAGE_SUFFIXES)

    def test_musubi_video_loader_capability_matches_the_pinned_source(self) -> None:
        self.assertEqual(MUSUBI_VIDEO_SUFFIXES, frozenset({
            ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg",
            ".webm", ".wmv",
        }))

    def test_frozen_suffixes_is_stable_and_compact(self) -> None:
        self.assertEqual(frozen_suffixes({".webm", ".mp4"}), '[".mp4",".webm"]')


if __name__ == "__main__":
    unittest.main()
