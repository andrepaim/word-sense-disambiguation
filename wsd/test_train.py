import json

import pytest
from transformers import AutoTokenizer, ModernBertConfig, ModernBertForMaskedLM

from wsd.letters import NUM_LETTERS

BASE_TOKENIZER = "answerdotai/ModernBERT-Large-Instruct"


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """A randomly initialized two-layer ModernBERT with the real tokenizer, so a training step runs in seconds."""
    path = tmp_path_factory.mktemp("tiny-modernbert")
    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    config = ModernBertConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                              num_hidden_layers=2, num_attention_heads=2)
    ModernBertForMaskedLM(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture
def generated_data_dir(tmp_path):
    """One word file in the format of training/data/generate.py."""
    data = tmp_path / "data"
    data.mkdir()
    synsets = [
        {"id": "omw-en-08420278-n", "pos": "n", "source_definition": "a financial institution",
         "alternative_definition": "a business that keeps and lends money",
         "examples": ["I went to the bank to deposit cash.", "The bank approved the loan."]},
        {"id": "omw-en-09213565-n", "pos": "n", "source_definition": "sloping land beside a body of water",
         "alternative_definition": "the ground along the edge of a river",
         "examples": ["We had a picnic on the bank of the river."]},
    ]
    (data / "bank.json").write_text(json.dumps(synsets))
    return data


def test_one_training_step_saves_a_pruned_model(tiny_model_dir, generated_data_dir, tmp_path):
    """Regression: main() used ``model.args`` (no such attribute) instead of ``model.config``."""
    from training.train import main

    final = main([
        "--model", str(tiny_model_dir), "--data-dir", str(generated_data_dir),
        "--output-dir", str(tmp_path / "out"), "--max-steps", "1", "--batch-size", "2",
        "--report-to", "none", "--eval-wn-count", "0", "--label-smoothing", "0.1",
    ])

    config = json.loads((final / "config.json").read_text())
    assert config["label_smoothing"] == 0.1  # read by WSDModernBertForMaskedLM.forward
    assert config["answer_vocab_size"] == NUM_LETTERS
    letters = json.loads((final / "answer_letters.json").read_text())
    assert letters["num_letters"] == NUM_LETTERS
