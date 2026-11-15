import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[3] / "base"
DOCKERFILE = (BASE / "Dockerfile").read_text(encoding="utf-8")


def test_image_ends_as_airflow_user():
    users = re.findall(r"^USER\s+(\S+)", DOCKERFILE, re.M)
    assert users[-1] == "airflow"


def test_no_transitional_libgl_package():
    assert "libgl1-mesa-glx" not in DOCKERFILE


def test_every_apt_install_skips_recommends():
    installs = re.findall(r"apt-get install[^\n\\]*", DOCKERFILE)
    assert installs and all("--no-install-recommends" in i for i in installs)


def test_pip_pins_airflow():
    assert re.search(r'pip install .*"apache-airflow==\$\{AIRFLOW_VERSION\}"', DOCKERFILE)


def test_requirements_pin_pyarrow():
    assert re.search(r"^pyarrow==\S+$", (BASE / "requirements.txt").read_text(), re.M)


def test_pip_run_uses_a_cache_mount():
    assert re.search(r"RUN --mount=type=cache,target=\S*pip\S*.*\n.*pip install", DOCKERFILE)


def test_requirements_do_not_pin_redis():
    # The base image's redis must match the Airflow redis provider.
    assert not re.search(r"^redis\b", (BASE / "requirements.txt").read_text(), re.M)


def test_dev_requirements_pin_each_package_once():
    root = BASE.parent.parent
    names = []
    for f in (root / "requirements-dev.txt", BASE / "requirements.txt"):
        names += re.findall(r"^([A-Za-z0-9_.-]+)==", f.read_text(), re.M)
    assert len(names) == len(set(names))
