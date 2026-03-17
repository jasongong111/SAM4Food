import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from configs.config import Config
import main
from main import parse_arguments, setup_config


def test_ingredient_mode_defaults_are_available():
    config = Config()

    assert config.model.use_ingredient_head is False
    assert config.model.num_ingredient_classes == 103
    assert config.model.ingredient_head_hidden_dim == 256
    assert config.training.classification_loss_weight == 1.0
    assert config.training.mask_loss_weight == 1.0
    assert config.data.dataset_name == "FoodSeg103"
    assert config.inference.use_prompted_aggregation is True
    assert config.inference.aggregation_score_threshold == 0.5
    assert config.inference.aggregation_iou_threshold == 0.5
    assert config.inference.max_prompts_per_image == 64


def test_cli_exposes_explicit_checkpoints_and_ingredient_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--sam_checkpoint",
            "sam_vit_b.pth",
            "--trained_checkpoint",
            "trained_model.pth",
            "--use_ingredient_head",
            "--num_ingredient_classes",
            "12",
            "--classification_loss_weight",
            "2.5",
            "--aggregation_score_threshold",
            "0.7",
        ],
    )

    args = parse_arguments()

    assert args.sam_checkpoint == "sam_vit_b.pth"
    assert args.trained_checkpoint == "trained_model.pth"
    assert args.use_ingredient_head is True
    assert args.num_ingredient_classes == 12
    assert args.classification_loss_weight == 2.5
    assert args.aggregation_score_threshold == 0.7

    config = setup_config(args)

    assert config.model.sam_checkpoint_path == "sam_vit_b.pth"
    assert config.model.trained_checkpoint_path == "trained_model.pth"
    assert config.model.use_ingredient_head is True
    assert config.model.num_ingredient_classes == 12
    assert config.training.classification_loss_weight == 2.5
    assert config.inference.aggregation_score_threshold == 0.7


def test_legacy_model_path_remains_compatible_for_eval(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--mode",
            "eval",
            "--model_path",
            "legacy_checkpoint.pth",
        ],
    )

    args = parse_arguments()
    config = setup_config(args)

    assert args.sam_checkpoint == "legacy_checkpoint.pth"
    assert args.trained_checkpoint == "legacy_checkpoint.pth"
    assert config.model.sam_checkpoint_path == "legacy_checkpoint.pth"
    assert config.model.trained_checkpoint_path == "legacy_checkpoint.pth"


def test_full_pipeline_uses_trained_checkpoint_for_eval_and_visualize(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    best_checkpoint = checkpoint_dir / "best_model.pth"
    best_checkpoint.write_text("checkpoint")

    config = Config()
    config.system.model_save_dir = str(checkpoint_dir)
    config.system.output_dir = str(tmp_path / "outputs")
    config.system.visualization_dir = str(tmp_path / "visualizations")

    args = SimpleNamespace(trained_checkpoint=None)
    calls = []

    trainer = SimpleNamespace(
        best_miou=0.75,
        model=SimpleNamespace(
            get_trainable_parameters=lambda: [torch.zeros(1)],
            parameters=lambda: [torch.zeros(2)],
        ),
    )
    history = {"train_history": [1, 2, 3]}

    def fake_train_model(config_arg, args_arg):
        return trainer, history

    def fake_evaluate_model_script(config_arg, args_arg):
        calls.append(("eval", args_arg.trained_checkpoint, config_arg.model.trained_checkpoint_path))
        return {"miou": 0.5}

    def fake_visualize_results(config_arg, args_arg):
        calls.append(("visualize", args_arg.trained_checkpoint, config_arg.model.trained_checkpoint_path))
        return "viz-dir"

    class FakeVisualizer:
        def __init__(self, config_arg, output_dir):
            self.output_dir = output_dir

        def create_training_analysis_plots(self, history_arg):
            return None

        def generate_evaluation_report(self, results_arg, comparison_arg):
            return "report"

    monkeypatch.setattr(main, "train_model", fake_train_model)
    monkeypatch.setattr(main, "evaluate_model_script", fake_evaluate_model_script)
    monkeypatch.setattr(main, "visualize_results", fake_visualize_results)
    monkeypatch.setattr(main, "Visualizer", FakeVisualizer)

    main.full_pipeline(config, args)

    expected_checkpoint = str(best_checkpoint)
    assert calls == [
        ("eval", expected_checkpoint, expected_checkpoint),
        ("visualize", expected_checkpoint, expected_checkpoint),
    ]
