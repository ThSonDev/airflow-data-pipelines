from unittest.mock import MagicMock

import consumer_postgres_streaming as consumer
import pytest


def test_stops_the_query_when_time_is_up():
    query = MagicMock()
    query.awaitTermination.return_value = False
    consumer.run_for(query, 5)
    query.awaitTermination.assert_called_once_with(5)
    query.stop.assert_called_once()


def test_does_not_stop_a_query_that_ended_by_itself():
    query = MagicMock()
    query.awaitTermination.return_value = True
    consumer.run_for(query, 5)
    query.stop.assert_not_called()


def test_a_failed_query_still_raises():
    query = MagicMock()
    query.awaitTermination.side_effect = RuntimeError("query failed")
    with pytest.raises(RuntimeError, match="query failed"):
        consumer.run_for(query, 5)
