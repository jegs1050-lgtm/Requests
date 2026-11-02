import pytest

import requests.adapters


def test_request_url_handles_leading_path_separators():
    """See also https://github.com/psf/requests/issues/6643."""
    a = requests.adapters.HTTPAdapter()
    p = requests.Request(method="GET", url="http://127.0.0.1:10000//v:h").prepare()
    assert "//v:h" == a.request_url(p, {})


def test_default_retry_policy_only_retries_connection_errors():
    retry = requests.adapters.HTTPAdapter().max_retries
    assert retry.total == requests.adapters.DEFAULT_RETRIES == 3
    assert retry.connect == 3
    assert retry.read is False
    assert retry.other == 0
    assert retry.respect_retry_after_header is False
    assert retry.backoff_factor == requests.adapters.DEFAULT_RETRY_BACKOFF


def test_max_retries_zero_disables_retries():
    retry = requests.adapters.HTTPAdapter(max_retries=0).max_retries
    assert retry.total == 0
    assert retry.read is False


def test_explicit_max_retries_keeps_urllib3_semantics():
    retry = requests.adapters.HTTPAdapter(max_retries=5).max_retries
    assert retry.total == 5
    assert retry.read is None


def test_explicit_retry_object_is_used_as_is():
    from urllib3.util import Retry

    custom = Retry(total=7, status_forcelist=[503])
    assert requests.adapters.HTTPAdapter(max_retries=custom).max_retries is custom


def test_connection_errors_are_retried_by_default(monkeypatch):
    """A refused connection is attempted 1 + DEFAULT_RETRIES times."""
    import socket

    from urllib3.connection import HTTPConnection

    # Grab a free port and close it so connections to it are refused.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    attempts = []
    original = HTTPConnection._new_conn

    def counting_new_conn(self):
        attempts.append(1)
        return original(self)

    monkeypatch.setattr(HTTPConnection, "_new_conn", counting_new_conn)
    monkeypatch.setattr(requests.adapters, "DEFAULT_RETRY_BACKOFF", 0)

    with pytest.raises(requests.exceptions.ConnectionError):
        requests.get(f"http://127.0.0.1:{port}/")
    assert len(attempts) == 1 + requests.adapters.DEFAULT_RETRIES
