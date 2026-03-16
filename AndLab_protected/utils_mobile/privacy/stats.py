"""
Statistics and persistence mixin for the Privacy Protection Layer.

Provides recording of anonymization statistics, saving/loading token mappings,
and summary generation for evaluation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


class StatsMixin:
    """
    Mixin that provides statistics recording and token mapping persistence.

    Expects the host class to have these instance attributes:
    - enabled: bool
    - _anonymization_stats: List[Dict[str, Any]]
    - _task_dir: Optional[str]
    - token_to_real: Dict[str, str]
    - real_to_token: Dict[str, str]
    - real_to_entity_type: Dict[str, str]
    - whitelist: set
    """

    def _record_statistics(self, type: str, original_length: int, anonymized_chars_count: int, num_tokens: int):
        """
        Record anonymization statistics.
        
        Args:
            type: Type of anonymization ("text", "xml", or "screenshot")
            original_length: Original text length
            anonymized_chars_count: Length of original characters that were anonymized (not the anonymized text length)
            num_tokens: Number of new tokens created
        """
        import time
        self._anonymization_stats.append({
            "type": type,
            "original_length": original_length,
            "anonymized_chars_count": anonymized_chars_count,
            "num_tokens": num_tokens,
            "timestamp": time.time()
        })

    def set_task_dir(self, task_dir: str):
        """
        Set the task directory for saving statistics.
        
        Args:
            task_dir: Path to the task directory (e.g., "logs/evaluation/20251214_1")
        """
        if not task_dir:
            self._task_dir = None
            return

        normalized = os.path.abspath(task_dir)
        try:
            os.makedirs(normalized, exist_ok=True)
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"[PrivacyProtection] Warning: Failed to create task dir {normalized}: {exc}")
        self._task_dir = normalized

    def save_stats(self):
        """
        Save anonymization statistics to a JSON file in the task directory.
        Also saves token mappings for later evaluation.
        """
        if not self._task_dir:
            return
        
        if self._anonymization_stats:
            stats_file = os.path.join(self._task_dir, "privacy_anonymization_stats.json")
            try:
                stats_to_save = self._anonymization_stats.copy()
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "task_dir": self._task_dir,
                        "total_records": len(stats_to_save),
                        "records": stats_to_save
                    }, f, ensure_ascii=False, indent=2)
                print(f"[PrivacyProtection] Statistics saved to {stats_file}")
            except Exception as e:
                print(f"[PrivacyProtection] Failed to save statistics: {e}")
            finally:
                self._anonymization_stats.clear()
        
        if self.enabled and self.token_to_real:
            self.save_token_mapping()
            self.token_to_real.clear()
            self.real_to_token.clear()
            self.real_to_entity_type.clear()
            self.whitelist.clear()

    def save_token_mapping(self):
        """
        Save token-to-real mapping to a JSON file in the task directory.
        This is important for evaluation where we need to convert anonymized tokens
        back to real values for comparison with golden answers.
        Also saves entity type information.
        """
        if not self._task_dir:
            return
        
        mapping_file = os.path.join(self._task_dir, "privacy_token_mapping.json")
        try:
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "task_dir": self._task_dir,
                    "token_to_real": self.token_to_real,
                    "real_to_token": self.real_to_token,
                    "real_to_entity_type": self.real_to_entity_type
                }, f, ensure_ascii=False, indent=2)
            print(f"[PrivacyProtection] Token mapping saved to {mapping_file}")
        except Exception as e:
            print(f"[PrivacyProtection] Failed to save token mapping: {e}")

    def load_token_mapping(self, task_dir: str):
        """
        Load token-to-real mapping from a JSON file in the task directory.
        This is used during evaluation to convert anonymized tokens back to real values.
        Also loads entity type information.
        
        Args:
            task_dir: Path to the task directory containing the mapping file.
        
        Returns:
            True if mapping was loaded successfully, False otherwise.
        """
        if not self.enabled:
            return False
        
        mapping_file = os.path.join(task_dir, "privacy_token_mapping.json")
        if not os.path.exists(mapping_file):
            return False
        
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.token_to_real = data.get("token_to_real", {})
                self.real_to_token = data.get("real_to_token", {})
                self.real_to_entity_type = data.get("real_to_entity_type", {})
            print(f"[PrivacyProtection] Token mapping loaded from {mapping_file} ({len(self.token_to_real)} tokens)")
            return True
        except Exception as e:
            print(f"[PrivacyProtection] Failed to load token mapping: {e}")
            return False

    def get_stats_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for anonymization.
        
        Returns:
            Dictionary with summary statistics including:
            - total_original_length: Total length of all original texts
            - total_anonymized_chars_count: Total length of original characters that were anonymized
            - anonymization_ratio: Percentage of anonymized characters to original length
            - total_records: Total number of anonymization operations
            - by_type: Statistics grouped by type (text, xml, screenshot)
        """
        if not self._anonymization_stats:
            return {
                "total_original_length": 0,
                "total_anonymized_chars_count": 0,
                "anonymization_ratio": 0.0,
                "total_records": 0,
                "by_type": {}
            }
        
        total_original = sum(stat["original_length"] for stat in self._anonymization_stats)
        total_anonymized_chars = sum(
            stat.get("anonymized_chars_count", stat.get("anonymized_length", 0)) 
            for stat in self._anonymization_stats
        )
        
        by_type: Dict[str, Dict[str, Any]] = {}
        for stat in self._anonymization_stats:
            stat_type = stat["type"]
            if stat_type not in by_type:
                by_type[stat_type] = {
                    "count": 0,
                    "original_length": 0,
                    "anonymized_chars_count": 0
                }
            by_type[stat_type]["count"] += 1
            by_type[stat_type]["original_length"] += stat["original_length"]
            anonymized_count = stat.get("anonymized_chars_count", stat.get("anonymized_length", 0))
            by_type[stat_type]["anonymized_chars_count"] += anonymized_count
        
        for stat_type in by_type:
            if by_type[stat_type]["original_length"] > 0:
                by_type[stat_type]["anonymization_ratio"] = (
                    by_type[stat_type]["anonymized_chars_count"] / by_type[stat_type]["original_length"] * 100
                )
            else:
                by_type[stat_type]["anonymization_ratio"] = 0.0
        
        anonymization_ratio = (total_anonymized_chars / total_original * 100) if total_original > 0 else 0.0
        
        return {
            "total_original_length": total_original,
            "total_anonymized_chars_count": total_anonymized_chars,
            "anonymization_ratio": anonymization_ratio,
            "total_records": len(self._anonymization_stats),
            "by_type": by_type
        }
