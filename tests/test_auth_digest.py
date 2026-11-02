"""Unit tests for HTTPDigestAuth's qop handling, including qop="auth-int"."""

import hashlib
import io

import pytest

from requests.auth import HTTPDigestAuth, _body_bytes_for_digest
from requests.utils import parse_dict_header

URL = "http://example.com/dir/index.html?x=1"
URI = "/dir/index.html?x=1"


def md5(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.md5(data).hexdigest()


def make_auth(qop, *, prefer_auth_int=False, algorithm=None):
    auth = HTTPDigestAuth("Mufasa", "Circle Of Life", prefer_auth_int=prefer_auth_int)
    auth.init_per_thread_state()
    chal = {
        "realm": "testrealm@host.com",
        "nonce": "dcd98b7102dd2f0e8b11d0f600bfb0c093",
    }
    if qop is not None:
        chal["qop"] = qop
    if algorithm is not None:
        chal["algorithm"] = algorithm
    auth._thread_local.chal = chal
    return auth


def parse(header):
    assert header.startswith("Digest ")
    return parse_dict_header(header[len("Digest ") :])


def expected_response(fields, method, body=None):
    """Recompute the response digest straight from RFC 7616, section 3.4.1."""
    ha1 = md5("Mufasa:testrealm@host.com:Circle Of Life")
    if fields.get("qop") == "auth-int":
        ha2 = md5(f"{method}:{URI}:{md5(body)}")
    else:
        ha2 = md5(f"{method}:{URI}")
    if "qop" not in fields:
        return md5(f"{ha1}:{fields['nonce']}:{ha2}")
    return md5(
        f"{ha1}:{fields['nonce']}:{fields['nc']}:{fields['cnonce']}:"
        f"{fields['qop']}:{ha2}"
    )


class TestQopChoice:
    def test_auth_only(self):
        fields = parse(make_auth("auth").build_digest_header("GET", URL))
        assert fields["qop"] == "auth"
        assert fields["response"] == expected_response(fields, "GET")

    def test_auth_int_only(self):
        body = b'{"lion": "king"}'
        header = make_auth("auth-int").build_digest_header("POST", URL, body)
        fields = parse(header)
        assert fields["qop"] == "auth-int"
        assert fields["response"] == expected_response(fields, "POST", body)

    def test_auth_int_body_changes_response(self):
        auth = make_auth("auth-int")
        a = parse(auth.build_digest_header("POST", URL, b"one"))
        b = parse(auth.build_digest_header("POST", URL, b"two"))
        assert a["response"] == expected_response(a, "POST", b"one")
        assert b["response"] == expected_response(b, "POST", b"two")
        assert expected_response(b, "POST", b"one") != b["response"]

    def test_auth_int_without_body_hashes_empty_string(self):
        fields = parse(make_auth("auth-int").build_digest_header("GET", URL))
        assert fields["response"] == expected_response(fields, "GET", b"")

    @pytest.mark.parametrize(
        "qop", ("auth,auth-int", "auth, auth-int", "auth-int, auth")
    )
    def test_auth_preferred_when_both_offered(self, qop):
        # "auth-int, auth" used to fail because " auth" wasn't stripped.
        fields = parse(make_auth(qop).build_digest_header("GET", URL))
        assert fields["qop"] == "auth"
        assert fields["response"] == expected_response(fields, "GET")

    def test_prefer_auth_int_when_both_offered(self):
        auth = make_auth("auth, auth-int", prefer_auth_int=True)
        fields = parse(auth.build_digest_header("PUT", URL, b"data"))
        assert fields["qop"] == "auth-int"
        assert fields["response"] == expected_response(fields, "PUT", b"data")

    def test_prefer_auth_int_falls_back_for_unreadable_body(self):
        auth = make_auth("auth, auth-int", prefer_auth_int=True)
        body = (chunk for chunk in [b"a", b"b"])
        fields = parse(auth.build_digest_header("POST", URL, body))
        assert fields["qop"] == "auth"

    def test_auth_int_only_with_unreadable_body_gives_up(self):
        body = (chunk for chunk in [b"a", b"b"])
        assert make_auth("auth-int").build_digest_header("POST", URL, body) is None

    def test_unknown_qop_gives_up(self):
        assert make_auth("token").build_digest_header("GET", URL) is None

    def test_no_qop_rfc2069(self):
        fields = parse(make_auth(None).build_digest_header("GET", URL))
        assert "qop" not in fields
        assert fields["response"] == expected_response(fields, "GET")

    def test_qop_is_case_insensitive(self):
        fields = parse(make_auth("AUTH-INT").build_digest_header("POST", URL, b"x"))
        assert fields["qop"] == "auth-int"


class TestBodyBytes:
    def test_none(self):
        assert _body_bytes_for_digest(None) == b""

    def test_bytes_and_bytearray(self):
        assert _body_bytes_for_digest(b"abc") == b"abc"
        assert _body_bytes_for_digest(bytearray(b"abc")) == b"abc"

    def test_str_is_encoded_like_the_wire(self):
        from requests.compat import is_urllib3_1

        expected = "café".encode("iso-8859-1" if is_urllib3_1 else "utf-8")
        assert _body_bytes_for_digest("café") == expected

    def test_seekable_file_is_rewound(self):
        f = io.BytesIO(b"0123456789")
        f.seek(3)
        assert _body_bytes_for_digest(f) == b"3456789"
        assert f.tell() == 3

    def test_text_file(self):
        assert _body_bytes_for_digest(io.StringIO("hi")) == b"hi"

    def test_generator_is_not_consumed(self):
        gen = (c for c in [b"a"])
        assert _body_bytes_for_digest(gen) is None
        assert next(gen) == b"a"


def test_auth_int_with_seekable_file_body_end_to_end():
    """handle_401 rewinds the file, so the digest covers the bytes resent."""
    auth = make_auth("auth-int")
    f = io.BytesIO(b"payload")
    header = auth.build_digest_header("POST", URL, f)
    fields = parse(header)
    assert fields["response"] == expected_response(fields, "POST", b"payload")
    assert f.read() == b"payload"


def test_prefer_auth_int_is_keyword_only():
    with pytest.raises(TypeError):
        HTTPDigestAuth("u", "p", True)
