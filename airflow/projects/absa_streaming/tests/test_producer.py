from types import SimpleNamespace

import producer
import pytest


class FakeProducer:
    leftover = 0

    def __init__(self, conf):
        pass

    def produce(self, topic, value, callback):
        pass

    def poll(self, t):
        pass

    def flush(self, t):
        return self.leftover


@pytest.fixture
def setup(monkeypatch, tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("review\nhello\nworld\n", encoding="utf-8")
    monkeypatch.setattr(producer, "CSV_PATH", str(csv))
    monkeypatch.setattr(producer, "failures", 0)
    monkeypatch.setattr(
        producer, "cfg", SimpleNamespace(kafka_bootstrap="k", kafka_topic="t", run_seconds=60, producer_delay=0),
    )
    monkeypatch.setattr(producer, "Producer", FakeProducer)
    monkeypatch.setattr(FakeProducer, "leftover", 0)


def test_clean_run_returns(setup):
    producer.main()


def test_unflushed_messages_exit_non_zero(setup, monkeypatch):
    monkeypatch.setattr(FakeProducer, "leftover", 2)
    with pytest.raises(SystemExit):
        producer.main()


def test_failed_delivery_exits_non_zero(setup):
    producer.delivery_report("boom", None)
    with pytest.raises(SystemExit):
        producer.main()
