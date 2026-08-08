import tempfile
import unittest
from pathlib import Path

from repositories import storage


class AnalysisHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_data_dir = storage.DATA_DIR
        self.original_history_dir = storage.ANALYSIS_HISTORY_DIR
        storage.DATA_DIR = Path(self.tempdir.name)
        storage.ANALYSIS_HISTORY_DIR = storage.DATA_DIR / 'analysis_history'

    def tearDown(self):
        storage.DATA_DIR = self.original_data_dir
        storage.ANALYSIS_HISTORY_DIR = self.original_history_dir
        self.tempdir.cleanup()

    def _snapshot(self, date, classification):
        return {
            'symbol': 'TEST',
            'data_date': date,
            'classification': classification,
            'setup_status': classification,
            'settings_fingerprint': 'profile-a',
            'ruleset_version': 'rules-v1',
            'checklist': [],
        }

    def test_snapshot_history_returns_current_data_date(self):
        self.assertEqual('2026-07-10', storage.record_analysis_snapshot(self._snapshot('2026-07-10', 'near_breakout')))
        self.assertEqual('2026-07-11', storage.record_analysis_snapshot(self._snapshot('2026-07-11', 'near_breakout')))
        self.assertEqual('2026-07-12', storage.record_analysis_snapshot(self._snapshot('2026-07-12', 'pullback_entry')))

    def test_settings_profiles_are_stored_as_separate_snapshots(self):
        storage.record_analysis_snapshot(self._snapshot('2026-07-10', 'near_breakout'))
        other_profile = self._snapshot('2026-07-11', 'near_breakout')
        other_profile['settings_fingerprint'] = 'profile-b'
        self.assertEqual('2026-07-11', storage.record_analysis_snapshot(other_profile))
