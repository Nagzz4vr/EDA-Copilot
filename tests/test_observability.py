import pytest
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Import the classes to test
from observability.trace_logger import TraceLogger
from observability.confidence_tracker import ConfidenceTracker, ConfidenceEvent
from observability.anomaly_detector import AnomalyDetector
from core.routing.confidence_router import ConfidenceRouter, ConfidenceTier


# ============================================================================
# TraceLogger Tests
# ============================================================================

class TestTraceLogger:
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test logs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def logger(self, temp_dir):
        """Create a TraceLogger instance with temporary directory"""
        return TraceLogger(session_id="test_session_123", log_dir=temp_dir)
    
    def test_initialization(self, temp_dir):
        """Test logger initialization creates directory and file path"""
        logger = TraceLogger(session_id="test_init", log_dir=temp_dir)
        
        assert logger.session_id == "test_init"
        assert logger.log_dir == Path(temp_dir)
        assert logger.log_dir.exists()
        assert logger.file_path == Path(temp_dir) / "test_init.jsonl"
    
    def test_log_creates_valid_entry(self, logger):
        """Test that log creates a valid JSON entry"""
        logger.log(
            tool="TEST_TOOL",
            intent="Testing log creation",
            inputs={"param1": "value1"},
            outputs={"result": "success"},
            confidence=0.95,
            usage_id=1
        )
        
        assert logger.file_path.exists()
        
        with open(logger.file_path, 'r') as f:
            line = f.readline()
            entry = json.loads(line)
        
        assert entry["schema_version"] == 1
        assert entry["session"] == "test_session_123"
        assert entry["tool"] == "TEST_TOOL"
        assert entry["intent"] == "Testing log creation"
        assert entry["inputs"] == {"param1": "value1"}
        assert entry["outputs"] == {"result": "success"}
        assert entry["confidence"] == 0.95
        assert entry["usage_ref"] == 1
        assert "ts" in entry
        assert "event_id" in entry
    
    def test_log_multiple_entries(self, logger):
        """Test logging multiple entries"""
        for i in range(5):
            logger.log(
                tool=f"TOOL_{i}",
                intent=f"Intent {i}",
                inputs={"index": i},
                outputs={"result": i * 2},
                confidence=0.5 + (i * 0.1)
            )
        
        entries = list(logger.stream())
        assert len(entries) == 5
        
        for i, entry in enumerate(entries):
            assert entry["tool"] == f"TOOL_{i}"
            assert entry["inputs"]["index"] == i
            assert entry["confidence"] == 0.5 + (i * 0.1)
    
    def test_event_id_uniqueness(self, logger):
        """Test that different entries get different event IDs"""
        logger.log(
            tool="TOOL_A",
            intent="First",
            inputs={},
            outputs={},
            confidence=0.8
        )
        
        logger.log(
            tool="TOOL_B",
            intent="Second",
            inputs={},
            outputs={},
            confidence=0.8
        )
        
        entries = list(logger.stream())
        assert len(entries) == 2
        assert entries[0]["event_id"] != entries[1]["event_id"]
    
    def test_event_id_consistency(self):
        """Test that identical entries get the same event ID"""
        logger1 = TraceLogger(session_id="session1", log_dir="/tmp/test1")
        logger2 = TraceLogger(session_id="session1", log_dir="/tmp/test2")
        
        entry = {
            "schema_version": 1,
            "tool": "TEST",
            "intent": "Test",
            "inputs": {},
            "outputs": {},
            "confidence": 0.8
        }
        
        id1 = logger1._make_event_id(entry)
        id2 = logger2._make_event_id(entry)
        
        assert id1 == id2
    
    def test_stream_empty_file(self, logger):
        """Test streaming from non-existent file"""
        entries = list(logger.stream())
        assert entries == []
    
    def test_stream_with_invalid_json(self, logger):
        """Test streaming skips invalid JSON lines"""
        # Manually write some invalid JSON
        with open(logger.file_path, 'w') as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')
        
        entries = list(logger.stream())
        assert len(entries) == 2
        assert entries[0]["valid"] == "json"
        assert entries[1]["another"] == "valid"
    
    def test_concurrent_logging(self, logger):
        """Test thread-safe concurrent logging"""
        import threading
        
        def log_entries(start_idx):
            for i in range(10):
                logger.log(
                    tool=f"TOOL_{start_idx}_{i}",
                    intent="Concurrent test",
                    inputs={"thread": start_idx, "index": i},
                    outputs={},
                    confidence=0.7
                )
        
        threads = [threading.Thread(target=log_entries, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        entries = list(logger.stream())
        assert len(entries) == 30


# ============================================================================
# ConfidenceTracker Tests
# ============================================================================

class TestConfidenceTracker:
    
    @pytest.fixture
    def tracker(self):
        """Create a ConfidenceTracker instance"""
        return ConfidenceTracker()
    
    def test_initialization(self, tracker):
        """Test tracker initializes with empty scores"""
        assert tracker.scores == {}
    
    def test_update_creates_new_uuid(self, tracker):
        """Test updating a new UUID creates entry"""
        tracker.update("uuid_1", "STEP_A", 0.85)
        
        assert "uuid_1" in tracker.scores
        assert len(tracker.scores["uuid_1"]) == 1
        assert tracker.scores["uuid_1"][0].step == "STEP_A"
        assert tracker.scores["uuid_1"][0].score == 0.85
    
    def test_update_appends_to_existing(self, tracker):
        """Test updating existing UUID appends new entry"""
        tracker.update("uuid_1", "STEP_A", 0.85)
        tracker.update("uuid_1", "STEP_B", 0.75)
        tracker.update("uuid_1", "STEP_C", 0.65)
        
        assert len(tracker.scores["uuid_1"]) == 3
        scores = [e.score for e in tracker.scores["uuid_1"]]
        assert scores == [0.85, 0.75, 0.65]
    
    def test_get_trend(self, tracker):
        """Test get_trend returns score list"""
        tracker.update("uuid_1", "STEP_A", 0.9)
        tracker.update("uuid_1", "STEP_B", 0.8)
        tracker.update("uuid_1", "STEP_C", 0.7)
        
        trend = tracker.get_trend("uuid_1")
        assert trend == [0.9, 0.8, 0.7]
    
    def test_get_trend_empty_uuid(self, tracker):
        """Test get_trend for non-existent UUID returns empty list"""
        trend = tracker.get_trend("nonexistent")
        assert trend == []
    
    def test_get_current_score(self, tracker):
        """Test get_current_score returns latest score"""
        tracker.update("uuid_1", "STEP_A", 0.9)
        tracker.update("uuid_1", "STEP_B", 0.8)
        tracker.update("uuid_1", "STEP_C", 0.7)
        
        current = tracker.get_current_score("uuid_1")
        assert current == 0.7
    
    def test_get_current_score_empty(self, tracker):
        """Test get_current_score for non-existent UUID returns None"""
        current = tracker.get_current_score("nonexistent")
        assert current is None
    
    def test_get_lowest_confidence_step(self, tracker):
        """Test get_lowest_confidence_step finds minimum"""
        tracker.update("uuid_1", "STEP_A", 0.9)
        tracker.update("uuid_1", "STEP_B", 0.3)  # Lowest
        tracker.update("uuid_1", "STEP_C", 0.7)
        
        lowest = tracker.get_lowest_confidence_step("uuid_1")
        assert lowest["step"] == "STEP_B"
        assert lowest["score"] == 0.3
        assert "timestamp" in lowest
    
    def test_get_lowest_confidence_step_empty(self, tracker):
        """Test get_lowest_confidence_step for empty UUID returns None"""
        lowest = tracker.get_lowest_confidence_step("nonexistent")
        assert lowest is None
    
    def test_get_score_drop(self, tracker):
        """Test get_score_drop calculates maximum drop"""
        tracker.update("uuid_1", "STEP_A", 0.9)
        tracker.update("uuid_1", "STEP_B", 0.8)  # Drop: 0.1
        tracker.update("uuid_1", "STEP_C", 0.4)  # Drop: 0.4 (max)
        tracker.update("uuid_1", "STEP_D", 0.5)  # Increase
        
        drop = tracker.get_score_drop("uuid_1")
        assert drop == 0.4
    
    def test_get_score_drop_no_drop(self, tracker):
        """Test get_score_drop returns None when scores only increase"""
        tracker.update("uuid_1", "STEP_A", 0.5)
        tracker.update("uuid_1", "STEP_B", 0.6)
        tracker.update("uuid_1", "STEP_C", 0.7)
        
        drop = tracker.get_score_drop("uuid_1")
        assert drop is None
    
    def test_get_score_drop_insufficient_data(self, tracker):
        """Test get_score_drop returns None with < 2 entries"""
        tracker.update("uuid_1", "STEP_A", 0.9)
        
        drop = tracker.get_score_drop("uuid_1")
        assert drop is None
    
    def test_ingest_from_trace_log(self, tracker):
        """Test ingesting confidence from trace log entry"""
        trace_entry = {
            "state_uuid": "uuid_1",
            "tool": "PARSER",
            "confidence": 0.88
        }
        
        tracker.ingest_from_trace_log(trace_entry)
        
        assert tracker.get_current_score("uuid_1") == 0.88
    
    def test_ingest_from_trace_log_session_fallback(self, tracker):
        """Test ingesting uses session if state_uuid missing"""
        trace_entry = {
            "session": "session_1",
            "tool": "VALIDATOR",
            "confidence": 0.75
        }
        
        tracker.ingest_from_trace_log(trace_entry)
        
        assert tracker.get_current_score("session_1") == 0.75
    
    def test_ingest_from_trace_log_missing_confidence(self, tracker):
        """Test ingesting skips entries without confidence"""
        trace_entry = {
            "state_uuid": "uuid_1",
            "tool": "TOOL"
        }
        
        tracker.ingest_from_trace_log(trace_entry)
        
        assert "uuid_1" not in tracker.scores
    
    def test_concurrent_updates(self, tracker):
        """Test thread-safe concurrent updates"""
        import threading
        
        def update_scores(uuid_suffix):
            for i in range(10):
                tracker.update(f"uuid_{uuid_suffix}", f"STEP_{i}", 0.5 + i * 0.01)
        
        threads = [threading.Thread(target=update_scores, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        for i in range(3):
            assert len(tracker.scores[f"uuid_{i}"]) == 10


# ============================================================================
# AnomalyDetector Tests
# ============================================================================

class TestAnomalyDetector:
    
    @pytest.fixture
    def tracker(self):
        """Create a ConfidenceTracker instance"""
        return ConfidenceTracker()
    
    @pytest.fixture
    def logger(tmp_path):
        return TraceLogger(session_id="test_session", log_dir="/tmp/test1")
    
    @pytest.fixture
    def detector(self, tracker,logger):
        """Create an AnomalyDetector instance"""
        return AnomalyDetector(tracker, logger)
    
    def test_initialization(self, detector):
        """Test detector initializes with empty escalations"""
        assert detector.escalated_uuids == set()
    
    def test_no_anomaly_insufficient_data(self, detector, tracker):
        """Test no anomaly with < 2 data points"""
        tracker.update("uuid_1", "STEP_A", 0.9)
        
        result = detector.check("uuid_1")
        assert result is False
    
    def test_large_drop_detection(self, detector, tracker):
        """Test detection of large confidence drop"""
        tracker.update("uuid_1", "STEP_A", 0.9)
        tracker.update("uuid_1", "STEP_B", 0.55)  # Drop of 0.35 > 0.30 threshold
        
        result = detector.check("uuid_1")
        assert result is True
        assert "uuid_1" in detector.escalated_uuids
    
    def test_large_drop_at_threshold(self, detector, tracker):
        """Test detection at exact threshold (0.30)"""
        tracker.update("uuid_1", "STEP_A", 0.8)
        tracker.update("uuid_1", "STEP_B", 0.5)  # Drop of exactly 0.30
        
        result = detector.check("uuid_1")
        assert result is True
    
    def test_no_anomaly_small_drop(self, detector, tracker):
        """Test no anomaly for small drop below threshold"""
        tracker.update("uuid_1", "STEP_A", 0.9)
        tracker.update("uuid_1", "STEP_B", 0.65)  # Drop of 0.25 < 0.30 threshold
        
        result = detector.check("uuid_1")
        assert result is False
    
    def test_floor_breach_detection(self, detector, tracker):
        """Test detection when confidence falls below floor"""
        tracker.update("uuid_1", "STEP_A", 0.8)
        tracker.update("uuid_1", "STEP_B", 0.35)  # Below 0.40 floor
        
        result = detector.check("uuid_1")
        assert result is True
        assert "uuid_1" in detector.escalated_uuids
    
    def test_floor_breach_at_threshold(self, detector, tracker):
        """Test no detection at exact floor value"""
        tracker.update("uuid_1", "STEP_A", 0.8)
        tracker.update("uuid_1", "STEP_B", 0.40)  # Exactly at floor
        
        result = detector.check("uuid_1")
        assert result is False
    
    def test_consecutive_low_scores_detection(self, detector, tracker):
        """Test detection of consecutive low scores"""
        tracker.update("uuid_1", "STEP_A", 0.45)
        tracker.update("uuid_1", "STEP_B", 0.48)  # Both < 0.5
        
        result = detector.check("uuid_1")
        assert result is True
        assert "uuid_1" in detector.escalated_uuids
    
    def test_consecutive_low_scores_with_high_score(self, detector, tracker):
        """Test no detection when one score is high"""
        tracker.update("uuid_1", "STEP_A", 0.45)
        tracker.update("uuid_1", "STEP_B", 0.65)  # Second is >= 0.5
        
        result = detector.check("uuid_1")
        assert result is False
    
    def test_consecutive_low_scores_three_values(self, detector, tracker):
        """Test consecutive low scores with more than 2 values"""
        tracker.update("uuid_1", "STEP_A", 0.55)
        tracker.update("uuid_1", "STEP_B", 0.45)
        tracker.update("uuid_1", "STEP_C", 0.48)  # Last 2 are low
        
        result = detector.check("uuid_1")
        assert result is True
    
    def test_no_repeat_escalation(self, detector, tracker):
        """Test that already escalated UUID doesn't trigger again"""
        tracker.update("uuid_1", "STEP_A", 0.9)
        tracker.update("uuid_1", "STEP_B", 0.55)  # Large drop
        
        # First check - should detect
        result1 = detector.check("uuid_1")
        assert result1 is True
        
        # Add more data with anomaly
        tracker.update("uuid_1", "STEP_C", 0.2)  # Below floor
        
        # Second check - should not detect (already escalated)
        result2 = detector.check("uuid_1")
        assert result2 is False
    
    def test_get_anomaly_reason_floor_breach(self, detector, tracker):
        """Test anomaly reason for floor breach"""
        tracker.update("uuid_1", "STEP_A", 0.8)
        tracker.update("uuid_1", "STEP_B", 0.35)
        
        reason = detector.get_anomaly_reason("uuid_1")
        assert "fell to 0.35" in reason
        assert "below floor" in reason
    
    def test_get_anomaly_reason_large_drop(self, detector, tracker):
        """Test anomaly reason for large drop"""
        tracker.update("uuid_1", "STEP_A", 0.9)
        tracker.update("uuid_1", "STEP_B", 0.55)
        
        reason = detector.get_anomaly_reason("uuid_1")
        assert "dropped" in reason
        assert "35.0%" in reason or "0.35" in reason
    
    def test_get_anomaly_reason_sustained_low(self, detector, tracker):
        """Test anomaly reason for sustained low confidence"""
        tracker.update("uuid_1", "STEP_A", 0.45)
        tracker.update("uuid_1", "STEP_B", 0.48)
        
        reason = detector.get_anomaly_reason("uuid_1")
        assert "Sustained low confidence" in reason
    
    def test_get_anomaly_reason_no_trend(self, detector, tracker):
        """Test anomaly reason with no trend data"""
        reason = detector.get_anomaly_reason("uuid_nonexistent")
        assert reason == "Unknown anomaly"
    
    def test_multiple_uuids(self, detector, tracker):
        """Test detector handles multiple UUIDs independently"""
        # UUID 1 - anomaly
        tracker.update("uuid_1", "STEP_A", 0.9)
        tracker.update("uuid_1", "STEP_B", 0.5)
        
        # UUID 2 - no anomaly
        tracker.update("uuid_2", "STEP_A", 0.8)
        tracker.update("uuid_2", "STEP_B", 0.75)
        
        result1 = detector.check("uuid_1")
        result2 = detector.check("uuid_2")
        
        assert result1 is True
        assert result2 is False
        assert "uuid_1" in detector.escalated_uuids
        assert "uuid_2" not in detector.escalated_uuids


# ============================================================================
# ConfidenceRouter Tests
# ============================================================================

class TestConfidenceRouter:
    
    def create_blob(self, signals: List[dict]):
        """Helper to create a mock blob with signals"""
        blob = Mock()
        blob.signals = signals
        return blob
    
    def test_classify_high_confidence(self):
        """Test classification of high confidence (>= 0.85)"""
        blob = self.create_blob([
            {"severity": "LOW", "missing_ratio": 0.05}
        ])
        
        tier = ConfidenceRouter.classify(blob)
        assert tier == ConfidenceTier.HIGH
    
    def test_classify_medium_confidence(self):
        """Test classification of medium confidence (0.60 - 0.84)"""
        blob = self.create_blob([
            {"severity": "MEDIUM", "missing_ratio": 0.15},
            {"severity": "LOW", "missing_ratio": 0.10}
        ])
        
        tier = ConfidenceRouter.classify(blob)
        assert tier == ConfidenceTier.MEDIUM
    
    def test_classify_low_confidence(self):
        """Test classification of low confidence (< 0.60)"""
        blob = self.create_blob([
            {"severity": "HIGH", "missing_ratio": 0.5},
            {"severity": "MEDIUM", "missing_ratio": 0.4}
        ])
        
        tier = ConfidenceRouter.classify(blob)
        assert tier == ConfidenceTier.LOW
    
    def test_leakage_forces_low_confidence(self):
        """Test that leakage flag forces low confidence"""
        blob = self.create_blob([
            {"severity": "LOW", "missing_ratio": 0.0, "leakage": True}
        ])
        
        tier = ConfidenceRouter.classify(blob)
        assert tier == ConfidenceTier.LOW
    
    def test_compute_score_no_signals(self):
        """Test score computation with no signals"""
        blob = self.create_blob([])
        
        score = ConfidenceRouter._compute_score(blob)
        assert score == 1.0  # No risk = full confidence
    
    def test_compute_score_low_severity(self):
        """Test score with low severity signals"""
        blob = self.create_blob([
            {"severity": "LOW", "missing_ratio": 0.0},
            {"severity": "LOW", "missing_ratio": 0.0}
        ])
        
        score = ConfidenceRouter._compute_score(blob)
        assert score > 0.8
    
    def test_compute_score_high_severity(self):
        """Test score with high severity signals"""
        blob = self.create_blob([
            {"severity": "CRITICAL", "missing_ratio": 0.0}
        ])
        
        score = ConfidenceRouter._compute_score(blob)
        assert score < 0.7
    
    def test_compute_score_high_missing_ratio(self):
        """Test score with high missing ratio"""
        blob = self.create_blob([
            {"severity": "LOW", "missing_ratio": 0.8},
            {"severity": "LOW", "missing_ratio": 0.9}
        ])
        
        score = ConfidenceRouter._compute_score(blob)
        assert score < 0.5
    
    def test_compute_score_many_signals(self):
        """Test score with many signals (count penalty)"""
        signals = [{"severity": "LOW", "missing_ratio": 0.0} for _ in range(25)]
        blob = self.create_blob(signals)
        
        score = ConfidenceRouter._compute_score(blob)
        # Should have count penalty applied
        assert score < 1.0
    
    def test_compute_score_combined_factors(self):
        """Test score with combined risk factors"""
        blob = self.create_blob([
            {"severity": "HIGH", "missing_ratio": 0.3},
            {"severity": "MEDIUM", "missing_ratio": 0.4},
            {"severity": "LOW", "missing_ratio": 0.2}
        ])
        
        score = ConfidenceRouter._compute_score(blob)
        assert 0.0 <= score <= 1.0
        assert score < 0.7  # Combined factors should lower score
    
    def test_compute_score_bounds(self):
        """Test score is always bounded between 0.0 and 1.0"""
        # Extreme risk case
        blob = self.create_blob([
            {"severity": "CRITICAL", "missing_ratio": 1.0, "leakage": False}
            for _ in range(30)
        ])
        
        score = ConfidenceRouter._compute_score(blob)
        assert 0.0 <= score <= 1.0
    
    def test_severity_scores_mapping(self):
        """Test severity score mappings are correct"""
        assert ConfidenceRouter.SEVERITY_SCORES["LOW"] == 0.2
        assert ConfidenceRouter.SEVERITY_SCORES["MEDIUM"] == 0.5
        assert ConfidenceRouter.SEVERITY_SCORES["HIGH"] == 0.8
        assert ConfidenceRouter.SEVERITY_SCORES["CRITICAL"] == 1.0
    
    def test_thresholds_configuration(self):
        """Test threshold configuration is correct"""
        assert ConfidenceRouter.THRESHOLDS["high"] == 0.85
        assert ConfidenceRouter.THRESHOLDS["medium"] == 0.60
        assert ConfidenceRouter.THRESHOLDS["low"] == 0.0
    
    def test_signal_without_severity(self):
        """Test handling signal without severity field"""
        blob = self.create_blob([
            {"missing_ratio": 0.1}  # No severity field
        ])
        
        score = ConfidenceRouter._compute_score(blob)
        assert 0.0 <= score <= 1.0
    
    def test_signal_without_missing_ratio(self):
        """Test handling signal without missing_ratio field"""
        blob = self.create_blob([
            {"severity": "MEDIUM"}  # No missing_ratio field
        ])
        
        score = ConfidenceRouter._compute_score(blob)
        assert 0.0 <= score <= 1.0
    
    def test_edge_case_exact_threshold_high(self):
        """Test exact threshold value for high confidence"""
        # Create blob that scores exactly 0.85
        blob = self.create_blob([
            {"severity": "LOW", "missing_ratio": 0.0375}
        ])
        
        score = ConfidenceRouter._compute_score(blob)
        tier = ConfidenceRouter.classify(blob)
        
        if abs(score - 0.85) < 0.01:  # If we hit the threshold
            assert tier == ConfidenceTier.HIGH
    
    def test_edge_case_exact_threshold_medium(self):
        """Test exact threshold value for medium confidence"""
        # Test value at exactly 0.60
        blob = self.create_blob([
            {"severity": "MEDIUM", "missing_ratio": 0.2}
        ])
        
        score = ConfidenceRouter._compute_score(blob)
        tier = ConfidenceRouter.classify(blob)
        
        if abs(score - 0.60) < 0.01:  # If we hit the threshold
            assert tier == ConfidenceTier.MEDIUM


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    
    def test_full_workflow(self, tmp_path):
        """Test complete workflow: Logger -> Tracker -> Detector"""
        # Setup
        logger = TraceLogger(session_id="integration_test", log_dir=str(tmp_path))
        tracker = ConfidenceTracker()
        detector = AnomalyDetector(tracker, logger)
        
        # Simulate workflow
        # Step 1: Normal operation
        logger.log(
            tool="PARSER",
            intent="Parse input",
            inputs={"data": "test"},
            outputs={"parsed": True},
            confidence=0.9,
            usage_id=1
        )
        
        # Ingest into tracker
        for entry in logger.stream():
            tracker.ingest_from_trace_log(entry)
        
        # Check - should be no anomaly
        assert detector.check("integration_test") is False
        
        # Step 2: Confidence drops
        logger.log(
            tool="VALIDATOR",
            intent="Validate",
            inputs={"parsed": True},
            outputs={"valid": False},
            confidence=0.55,  # Drop of 0.35
            usage_id=2
        )
        
        # Ingest and check
        entries = list(logger.stream())
        tracker.ingest_from_trace_log(entries[-1])
        
        # Should detect anomaly
        assert detector.check("integration_test") is True
        assert "integration_test" in detector.escalated_uuids
    
    def test_confidence_router_integration(self):
        """Test ConfidenceRouter with realistic blob"""
        blob = Mock()
        blob.signals = [
            {"severity": "MEDIUM", "missing_ratio": 0.15},
            {"severity": "LOW", "missing_ratio": 0.10},
            {"severity": "LOW", "missing_ratio": 0.05}
        ]
        
        tier = ConfidenceRouter.classify(blob)
        score = ConfidenceRouter._compute_score(blob)
        
        # Verify classification matches score
        if score >= 0.85:
            assert tier == ConfidenceTier.HIGH
        elif score >= 0.60:
            assert tier == ConfidenceTier.MEDIUM
        else:
            assert tier == ConfidenceTier.LOW


