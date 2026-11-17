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
    assert all("--no-install-recommends" in i for i in installs)


def test_base_is_slim_airflow_3_4():
    assert re.search(r"^FROM apache/airflow:slim-3\.4\.0-python3\.12$", DOCKERFILE, re.M)


def test_both_pip_steps_pin_airflow_and_the_first_uses_constraints():
    installs = re.findall(r"pip install (?:[^\n]*\\\n)*[^\n]*", DOCKERFILE)
    assert len(installs) == 2
    assert all(re.search(r'apache-airflow(\[[\w,]+\])?==\$\{AIRFLOW_VERSION\}', i) for i in installs)
    assert "--constraint" in installs[0] and "--constraint" not in installs[1]


def test_image_checks_dependencies():
    assert re.search(r"^RUN pip check$", DOCKERFILE, re.M)


def test_vehicle_libs_are_commented_out():
    text = (BASE / "requirements.txt").read_text()
    assert not re.search(r"^(ultralytics|opencv\S*|pillow)==", text, re.M | re.I)
    assert "libgl1" not in DOCKERFILE


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


STREAMLIT = BASE.parent / "projects" / "absa_streaming" / "streamlit"


def _pins(path):
    return dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)$", path.read_text(), re.M))


def test_streamlit_shared_pins_match_base_requirements():
    # Only the packages both images pin; the rest (streamlit, sqlalchemy, ...) belong to the dashboard image
    base = _pins(BASE / "requirements.txt")
    own = _pins(STREAMLIT / "requirements.txt")
    shared = base.keys() & own.keys()
    assert {"pandas", "numpy", "pyarrow"} <= shared
    assert all(base[n] == own[n] for n in shared)


def test_streamlit_image_is_slim_python_and_not_root():
    text = (STREAMLIT / "Dockerfile").read_text()
    assert re.search(r"^FROM python:3\.12-slim$", text, re.M)
    assert re.findall(r"^USER\s+(\S+)", text, re.M)[-1] != "root"
    # pip must run as root, otherwise streamlit lands in ~/.local/bin, which is not on PATH
    assert text.index("pip install") < text.rindex("USER ")


def test_java_17_is_copied_from_temurin_without_apt():
    copy = r"^COPY --from=eclipse-temurin:17\.\d+\.\d+_\d+-jre\S* /opt/java/openjdk /opt/java/openjdk$"
    assert re.search(copy, DOCKERFILE, re.M)
    assert "ENV JAVA_HOME=/opt/java/openjdk" in DOCKERFILE
    assert "openjdk-21" not in DOCKERFILE and "apt-get" not in DOCKERFILE


def test_requirements_include_setuptools_for_pyspark_distutils():
    assert re.search(r"^setuptools<81$", (BASE / "requirements.txt").read_text(), re.M)
