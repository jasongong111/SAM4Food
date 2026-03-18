import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for CI
import matplotlib.pyplot as plt
from src.utils.visualization import colorize_semantic_map, visualize_semantic_map


def test_colorize_background_is_black():
    sem_map = torch.zeros(8, 8, dtype=torch.long)
    color_map = colorize_semantic_map(sem_map, num_classes=5)
    assert color_map.shape == (8, 8, 3)
    assert (color_map == 0).all(), "Background (class 0) should be black"


def test_colorize_is_reproducible():
    sem_map = torch.tensor([[0, 1, 2], [3, 0, 1]], dtype=torch.long)
    a = colorize_semantic_map(sem_map, num_classes=5)
    b = colorize_semantic_map(sem_map, num_classes=5)
    np.testing.assert_array_equal(a, b)


def test_colorize_nonzero_classes_are_not_black():
    sem_map = torch.ones(8, 8, dtype=torch.long)
    color_map = colorize_semantic_map(sem_map, num_classes=5)
    # Class 1 should map to a non-black color
    assert not (color_map == 0).all(), "Class 1 should not be black"


def test_visualize_returns_figure():
    image_np = np.random.rand(16, 16, 3).astype(np.float32)
    sem_map = torch.zeros(16, 16, dtype=torch.long)
    sem_map[4:12, 4:12] = 1
    class_names = {1: "apple"}
    fig = visualize_semantic_map(image_np, sem_map, class_names)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_visualize_legend_colors_match_overlay():
    """Legend patch color for each class must match the pixel color in the overlay."""
    image_np = np.zeros((32, 32, 3), dtype=np.float32)
    sem_map = torch.zeros(32, 32, dtype=torch.long)
    sem_map[:16, :] = 1   # class 1 in top half
    sem_map[16:, :] = 2   # class 2 in bottom half
    class_names = {1: "carrot", 2: "broccoli"}
    fig = visualize_semantic_map(image_np, sem_map, class_names, num_classes=5)

    # Extract legend patch colors
    legend_colors = {}
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend is not None:
            for handle, text in zip(legend.legend_handles, legend.get_texts()):
                name = text.get_text()
                color_rgba = handle.get_facecolor()
                legend_colors[name] = np.array(color_rgba[:3])

    # Get overlay pixel colors from color_map
    color_map = colorize_semantic_map(sem_map, num_classes=5)
    for cid, name in class_names.items():
        if name in legend_colors:
            pixel_color = color_map[sem_map.numpy() == cid][0].astype(float) / 255.0
            np.testing.assert_allclose(
                legend_colors[name], pixel_color, atol=0.01,
                err_msg=f"Legend color for '{name}' (class {cid}) doesn't match overlay pixel color"
            )
    plt.close(fig)
