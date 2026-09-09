import os
import subprocess
import tempfile
import pytest
from pathlib import Path
from PIL import Image

# Path to the executable (assuming it's built in the 'build' directory)
# Adjust this based on whether it's run in Windows or WSL
EXEC_NAME = "gfpgan-ncnn-vulkan"
if os.name == 'nt':
    EXEC_NAME += ".exe"

def get_exec_path():
    # Try to find the executable in common build directories
    possible_paths = [
        Path(".") / "build" / "Release" / EXEC_NAME,
        Path(".") / "build" / "Debug" / EXEC_NAME,
        Path(".") / "build" / EXEC_NAME,
        Path(".") / EXEC_NAME,
    ]
    for p in possible_paths:
        if p.exists():
            return str(p.resolve())
    return None

@pytest.fixture
def dummy_image(tmp_path):
    """Creates a small dummy image for testing."""
    img_path = tmp_path / "dummy_input.jpg"
    img = Image.new('RGB', (100, 100), color = (73, 109, 137))
    img.save(img_path)
    return str(img_path)

@pytest.fixture
def dummy_models_dir(tmp_path):
    """Creates a mock models directory with empty files to pass basic checks."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "encoder.param").touch()
    (models_dir / "encoder.bin").touch()
    (models_dir / "style.bin").touch()
    (models_dir / "yolov5-blazeface.param").touch()
    (models_dir / "yolov5-blazeface.bin").touch()
    (models_dir / "real_esrgan.param").touch()
    (models_dir / "real_esrgan.bin").touch()
    return str(models_dir)

def test_executable_not_found():
    assert get_exec_path() is not None, f"Executable {EXEC_NAME} not found. Please build the project first."

def test_cli_help():
    exec_path = get_exec_path()
    if not exec_path:
        pytest.skip("Executable not built.")

    result = subprocess.run([exec_path, "-h"], capture_output=True, text=True)
    assert "Usage:" in result.stderr
    assert "-i input-path" in result.stderr
    assert "-o output-path" in result.stderr
    assert "-m model-path" in result.stderr
    assert "-g gpu-id" in result.stderr

def test_cli_missing_input():
    exec_path = get_exec_path()
    if not exec_path:
        pytest.skip("Executable not built.")

    result = subprocess.run([exec_path], capture_output=True, text=True)
    assert result.returncode == 1
    assert "Usage:" in result.stderr

def test_cli_custom_output(dummy_image, dummy_models_dir, tmp_path):
    exec_path = get_exec_path()
    if not exec_path:
        pytest.skip("Executable not built.")

    output_img = tmp_path / "custom_output.png"

    # We expect this might fail cleanly or succeed depending on whether the dummy models
    # are valid enough for ncnn. We are mostly testing if it *attempts* to read arguments.
    # Note: Since the models are empty 0-byte files, ncnn load will fail with:
    # "open param file ... failed" or "find_blob ... failed".
    # But it should parse the CLI arguments correctly first.

    result = subprocess.run([
        exec_path,
        "-i", dummy_image,
        "-o", str(output_img),
        "-m", dummy_models_dir,
        "-g", "0"
    ], capture_output=True, text=True)

    # Check that it tried to use our model path (meaning it parsed -m)
    # The exact error depends on ncnn, but it shouldn't be the CLI usage error
    assert "Usage:" not in result.stderr

    # If the real models were present, we would check: assert output_img.exists()
    # Since they are fake, we just verify it didn't fail at the CLI parsing step.
