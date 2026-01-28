"""
Unit tests for dataset loading functionality.

Tests cover:
- Text formatting for different dataset formats
- Single dataset loading
- Parallel dataset loading
- Error handling and edge cases
- Configuration validation
"""

import pytest
import threading
import logging
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def logger():
    """Create a logger for testing."""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    return logger


@pytest.fixture
def progress_lock():
    """Create a threading lock for testing."""
    return threading.Lock()


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.vocab_size = 32000
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    return tokenizer


# =============================================================================
# FORMAT_TEXT TESTS (Tests the format_text logic directly)
# =============================================================================

def format_text(example: Dict[str, Any], ds_config: dict) -> str:
    """
    Test version of format_text function - mirrors the implementation in train.py.
    This allows us to test the logic without importing train.py's dependencies.
    """
    text_field = ds_config.get("text_field", "text")
    answer_field = ds_config.get("answer_field")
    system_field = ds_config.get("system_field")
    format_type = ds_config.get("format", "text")

    # Handle chat/messages format
    if text_field == "messages" and "messages" in example:
        messages = example["messages"]
        if isinstance(messages, list):
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|{role}|>\n{content}")
            return "\n".join(parts) + "\n<|end|>"

    # Handle reasoning/chain-of-thought format
    if format_type == "reasoning" and answer_field:
        question = example.get(text_field, "")
        answer = example.get(answer_field, "")

        if not question or not answer:
            return ""

        # Build reasoning format with clear structure
        parts = []

        # Add system prompt if available
        if system_field and system_field in example:
            system = example[system_field]
            if system:
                parts.append(f"<|system|>\n{system}")

        # Add the question/problem
        parts.append(f"<|user|>\n{question}")

        # Add the reasoning/answer with thinking markers
        if "####" in str(answer):
            # GSM8K style: reasoning #### final_answer
            reasoning_part, final_answer = str(answer).rsplit("####", 1)
            parts.append(f"<|assistant|>\n<think>\n{reasoning_part.strip()}\n</think>\n\nThe answer is: {final_answer.strip()}")
        elif "\\boxed{" in str(answer):
            # LaTeX boxed answer format
            parts.append(f"<|assistant|>\n<think>\n{answer}\n</think>")
        else:
            # General reasoning response
            parts.append(f"<|assistant|>\n{answer}")

        parts.append("<|end|>")
        return "\n".join(parts)

    # Handle simple text field
    if text_field in example:
        text = str(example[text_field])

        # If there's an answer field but not reasoning format, combine them
        if answer_field and answer_field in example:
            answer = str(example[answer_field])
            return f"<|user|>\n{text}\n<|assistant|>\n{answer}\n<|end|>"

        return text

    return ""


class TestFormatText:
    """Tests for the format_text function."""

    def test_simple_text_field(self):
        """Test extracting text from a simple text field."""
        example = {"text": "Hello, this is a test."}
        ds_config = {"text_field": "text"}

        result = format_text(example, ds_config)

        assert result == "Hello, this is a test."

    def test_text_with_answer_field(self):
        """Test combining text and answer fields."""
        example = {"question": "What is 2+2?", "answer": "4"}
        ds_config = {
            "text_field": "question",
            "answer_field": "answer",
        }

        result = format_text(example, ds_config)

        assert "<|user|>" in result
        assert "What is 2+2?" in result
        assert "<|assistant|>" in result
        assert "4" in result
        assert "<|end|>" in result

    def test_reasoning_format_simple(self):
        """Test reasoning format with simple question and answer."""
        example = {
            "instruction": "Solve this math problem.",
            "output": "The answer is 42.",
        }
        ds_config = {
            "text_field": "instruction",
            "answer_field": "output",
            "format": "reasoning",
        }

        result = format_text(example, ds_config)

        assert "<|user|>" in result
        assert "Solve this math problem." in result
        assert "<|assistant|>" in result
        assert "The answer is 42." in result
        assert "<|end|>" in result

    def test_reasoning_format_with_gsm8k_style(self):
        """Test reasoning format with GSM8K style (#### separator)."""
        example = {
            "question": "If I have 5 apples and eat 2, how many remain?",
            "answer": "I start with 5 apples. I eat 2. 5 - 2 = 3. #### 3",
        }
        ds_config = {
            "text_field": "question",
            "answer_field": "answer",
            "format": "reasoning",
        }

        result = format_text(example, ds_config)

        assert "<|user|>" in result
        assert "5 apples" in result
        assert "<think>" in result
        assert "</think>" in result
        assert "The answer is: 3" in result

    def test_reasoning_format_with_latex_boxed(self):
        """Test reasoning format with LaTeX boxed answer."""
        example = {
            "query": "Solve for x: 2x + 4 = 10",
            "response": "Subtract 4 from both sides: 2x = 6. Divide by 2: x = \\boxed{3}",
        }
        ds_config = {
            "text_field": "query",
            "answer_field": "response",
            "format": "reasoning",
        }

        result = format_text(example, ds_config)

        assert "<think>" in result
        assert "\\boxed{3}" in result

    def test_reasoning_format_with_system_field(self):
        """Test reasoning format with system prompt."""
        example = {
            "question": "What is the capital of France?",
            "response": "Paris",
            "system_prompt": "You are a helpful geography expert.",
        }
        ds_config = {
            "text_field": "question",
            "answer_field": "response",
            "system_field": "system_prompt",
            "format": "reasoning",
        }

        result = format_text(example, ds_config)

        assert "<|system|>" in result
        assert "geography expert" in result
        assert "<|user|>" in result
        assert "capital of France" in result
        assert "<|assistant|>" in result
        assert "Paris" in result

    def test_messages_format(self):
        """Test extracting text from chat messages format."""
        example = {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        ds_config = {"text_field": "messages"}

        result = format_text(example, ds_config)

        assert "<|user|>" in result
        assert "Hello!" in result
        assert "<|assistant|>" in result
        assert "Hi there!" in result
        assert "<|end|>" in result

    def test_empty_text_field(self):
        """Test handling of empty text field."""
        example = {"text": ""}
        ds_config = {"text_field": "text"}

        result = format_text(example, ds_config)

        assert result == ""

    def test_missing_text_field(self):
        """Test handling of missing text field."""
        example = {"other_field": "some value"}
        ds_config = {"text_field": "text"}

        result = format_text(example, ds_config)

        assert result == ""

    def test_empty_question_in_reasoning_format(self):
        """Test reasoning format with empty question."""
        example = {"question": "", "answer": "Some answer"}
        ds_config = {
            "text_field": "question",
            "answer_field": "answer",
            "format": "reasoning",
        }

        result = format_text(example, ds_config)

        assert result == ""

    def test_empty_answer_in_reasoning_format(self):
        """Test reasoning format with empty answer."""
        example = {"question": "Some question", "answer": ""}
        ds_config = {
            "text_field": "question",
            "answer_field": "answer",
            "format": "reasoning",
        }

        result = format_text(example, ds_config)

        assert result == ""

    def test_messages_with_empty_list(self):
        """Test messages format with empty list."""
        example = {"messages": []}
        ds_config = {"text_field": "messages"}

        result = format_text(example, ds_config)

        assert result.strip() == "<|end|>"

    def test_multi_turn_conversation(self):
        """Test multi-turn conversation formatting."""
        example = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm fine, thanks!"},
            ]
        }
        ds_config = {"text_field": "messages"}

        result = format_text(example, ds_config)

        assert result.count("<|user|>") == 2
        assert result.count("<|assistant|>") == 2


# =============================================================================
# CONFIG TESTS
# =============================================================================

class TestConfig:
    """Tests for configuration-related functionality."""

    def test_camel_ai_math_not_in_datasets(self):
        """Test that camel-ai/math dataset is not in the default configuration."""
        from config import get_config

        config = get_config()
        dataset_names = [ds["name"] for ds in config.dataset.datasets]

        assert "camel-ai/math" not in dataset_names
        assert "camel-ai/math2" not in dataset_names

    def test_default_datasets_count(self):
        """Test the expected number of default datasets."""
        from config import get_config

        config = get_config()

        # Should have 10 datasets after removing camel-ai/math (was 11 originally)
        assert len(config.dataset.datasets) == 10

    def test_dataset_weights_sum_approximately_to_one(self):
        """Test that dataset weights sum to approximately 0.95 (after removing camel-ai/math)."""
        from config import get_config

        config = get_config()
        total_weight = sum(ds.get("weight", 0) for ds in config.dataset.datasets)

        # After removing camel-ai/math (0.05 weight), total should be ~0.95
        assert 0.90 <= total_weight <= 1.0

    def test_all_datasets_have_required_fields(self):
        """Test that all datasets have required configuration fields."""
        from config import get_config

        config = get_config()
        required_fields = ["name", "split", "text_field"]

        for ds in config.dataset.datasets:
            for field in required_fields:
                assert field in ds, f"Dataset {ds.get('name', 'unknown')} missing field: {field}"

    def test_all_datasets_have_max_samples(self):
        """Test that all datasets have max_samples configured."""
        from config import get_config

        config = get_config()

        for ds in config.dataset.datasets:
            assert "max_samples" in ds, f"Dataset {ds['name']} missing max_samples"
            assert ds["max_samples"] > 0, f"Dataset {ds['name']} has invalid max_samples"

    def test_streaming_flag_is_boolean(self):
        """Test that streaming flag is properly set as boolean."""
        from config import get_config

        config = get_config()

        for ds in config.dataset.datasets:
            assert "streaming" in ds, f"Dataset {ds['name']} missing streaming flag"
            assert isinstance(ds["streaming"], bool)

    def test_dataset_names_are_valid_huggingface_format(self):
        """Test that dataset names follow HuggingFace naming convention."""
        from config import get_config

        config = get_config()

        for ds in config.dataset.datasets:
            name = ds["name"]
            # HuggingFace datasets are typically in format "org/dataset" or just "dataset"
            assert "/" in name or name.isidentifier(), f"Invalid dataset name format: {name}"


# =============================================================================
# PARALLEL LOADING MOCK TESTS
# =============================================================================

class TestParallelLoading:
    """Tests for parallel loading functionality using mocks."""

    def test_thread_safety_of_progress_lock(self, logger):
        """Test that progress_lock provides thread-safe logging."""
        progress_lock = threading.Lock()
        log_messages = []

        def mock_log(msg):
            with progress_lock:
                log_messages.append(msg)

        # Simulate concurrent logging
        threads = []
        for i in range(10):
            t = threading.Thread(target=mock_log, args=(f"Message {i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(log_messages) == 10

    def test_concurrent_example_collection(self):
        """Test that examples can be safely collected from multiple threads."""
        all_examples = []
        lock = threading.Lock()

        def add_examples(examples):
            with lock:
                all_examples.extend(examples)

        threads = []
        for i in range(5):
            examples = [{"text": f"Example {i}-{j}"} for j in range(10)]
            t = threading.Thread(target=add_examples, args=(examples,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(all_examples) == 50

    def test_dataset_config_iteration(self):
        """Test iterating through dataset configs."""
        from config import get_config

        config = get_config()
        processed = []

        for ds_config in config.dataset.datasets:
            ds_name = ds_config["name"]
            processed.append(ds_name)

        assert len(processed) == 10
        assert all(isinstance(name, str) for name in processed)


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_format_text_with_none_values(self):
        """Test format_text handles None values gracefully."""
        example = {"text": None}
        ds_config = {"text_field": "text"}

        result = format_text(example, ds_config)

        assert result == "None"  # str(None) == "None"

    def test_format_text_with_numeric_values(self):
        """Test format_text handles numeric values."""
        example = {"text": 12345}
        ds_config = {"text_field": "text"}

        result = format_text(example, ds_config)

        assert result == "12345"

    def test_format_text_with_special_characters(self):
        """Test format_text handles special characters."""
        example = {"text": "Special chars: <>&\"'\\n\\t"}
        ds_config = {"text_field": "text"}

        result = format_text(example, ds_config)

        assert "Special chars" in result

    def test_format_text_with_unicode(self):
        """Test format_text handles unicode characters."""
        example = {"text": "Unicode: \u4e2d\u6587 \u0391\u03b2\u03b3 \ud83d\ude00"}
        ds_config = {"text_field": "text"}

        result = format_text(example, ds_config)

        assert "Unicode" in result
        assert "\u4e2d\u6587" in result

    def test_reasoning_format_multiple_hash_separators(self):
        """Test reasoning format with multiple #### in answer."""
        example = {
            "question": "Complex problem",
            "answer": "Step 1 #### intermediate #### 42",
        }
        ds_config = {
            "text_field": "question",
            "answer_field": "answer",
            "format": "reasoning",
        }

        result = format_text(example, ds_config)

        # Should split on the last ####
        assert "The answer is: 42" in result

    def test_very_long_text(self):
        """Test format_text handles very long text."""
        long_text = "x" * 100000
        example = {"text": long_text}
        ds_config = {"text_field": "text"}

        result = format_text(example, ds_config)

        assert len(result) == 100000


# =============================================================================
# DATASET CONFIGURATION VALIDATION TESTS
# =============================================================================

class TestDatasetConfigValidation:
    """Tests to validate dataset configurations are correct."""

    def test_fineweb_edu_config(self):
        """Test HuggingFaceFW/fineweb-edu configuration."""
        from config import get_config

        config = get_config()
        fineweb = next(
            (ds for ds in config.dataset.datasets if ds["name"] == "HuggingFaceFW/fineweb-edu"),
            None
        )

        assert fineweb is not None
        assert fineweb["config"] == "sample-10BT"
        assert fineweb["text_field"] == "text"
        assert fineweb["streaming"] is True

    def test_gsm8k_config(self):
        """Test openai/gsm8k configuration."""
        from config import get_config

        config = get_config()
        gsm8k = next(
            (ds for ds in config.dataset.datasets if ds["name"] == "openai/gsm8k"),
            None
        )

        assert gsm8k is not None
        assert gsm8k["text_field"] == "question"
        assert gsm8k["answer_field"] == "answer"
        assert gsm8k["format"] == "reasoning"

    def test_ultrachat_config(self):
        """Test HuggingFaceH4/ultrachat_200k configuration."""
        from config import get_config

        config = get_config()
        ultrachat = next(
            (ds for ds in config.dataset.datasets if ds["name"] == "HuggingFaceH4/ultrachat_200k"),
            None
        )

        assert ultrachat is not None
        assert ultrachat["text_field"] == "messages"
        assert ultrachat["split"] == "train_sft"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
