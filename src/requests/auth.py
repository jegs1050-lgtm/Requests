"""
requests.auth
~~~~~~~~~~~~~

This module contains the authentication handlers for Requests.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
import warnings
from base64 import b64encode
from typing import TYPE_CHECKING, Any, Final, cast, overload

from ._internal_utils import to_native_string
from .compat import basestring, is_urllib3_1, str, urlparse
from .cookies import extract_cookies_to_jar
from .utils import parse_dict_header

if TYPE_CHECKING:
    from http.cookiejar import CookieJar
    from typing import Any

    from .models import PreparedRequest, Response

CONTENT_TYPE_FORM_URLENCODED: Final = "application/x-www-form-urlencoded"
CONTENT_TYPE_MULTI_PART: Final = "multipart/form-data"


def _basic_auth_str(username: bytes | str, password: bytes | str) -> str:
    """Returns a Basic Auth string."""

    # "I want us to put a big-ol' comment on top of it that
    # says that this behaviour is dumb but we need to preserve
    # it because people are relying on it."
    #    - Lukasa
    #
    # These are here solely to maintain backwards compatibility
    # for things like ints. This will be removed in 3.0.0.
    if not isinstance(username, basestring):  # runtime guard for non-str/bytes
        warnings.warn(
            "Non-string usernames will no longer be supported in Requests "
            f"3.0.0. Please convert the object you've passed in ({username!r}) to "
            "a string or bytes object in the near future to avoid "
            "problems.",
            category=DeprecationWarning,
        )
        username = str(username)

    if not isinstance(password, basestring):  # runtime guard for non-str/bytes
        warnings.warn(
            "Non-string passwords will no longer be supported in Requests "
            f"3.0.0. Please convert the object you've passed in ({type(password)!r}) to "
            "a string or bytes object in the near future to avoid "
            "problems.",
            category=DeprecationWarning,
        )
        password = str(password)
    # -- End Removal --

    if isinstance(username, str):
        username = username.encode("latin1")

    if isinstance(password, str):
        password = password.encode("latin1")

    authstr = "Basic " + to_native_string(
        b64encode(b":".join((username, password))).strip()
    )

    return authstr


class AuthBase:
    """Base class that all auth implementations derive from"""

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        raise NotImplementedError("Auth hooks must be callable.")


class HTTPBasicAuth(AuthBase):
    """Attaches HTTP Basic Authentication to the given Request object."""

    username: bytes | str
    password: bytes | str

    @overload
    def __init__(self, username: str, password: str) -> None: ...
    @overload
    def __init__(self, username: bytes, password: bytes) -> None: ...

    def __init__(self, username: bytes | str, password: bytes | str) -> None:
        self.username = username
        self.password = password

    def __eq__(self, other: object) -> bool:
        return all(
            [
                self.username == getattr(other, "username", None),
                self.password == getattr(other, "password", None),
            ]
        )

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        r.headers["Authorization"] = _basic_auth_str(self.username, self.password)
        return r


class HTTPProxyAuth(HTTPBasicAuth):
    """Attaches HTTP Proxy Authentication to a given Request object."""

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        r.headers["Proxy-Authorization"] = _basic_auth_str(self.username, self.password)
        return r


def _body_bytes_for_digest(body: Any) -> bytes | None:
    """Return the exact bytes that will be sent as the request body, for
    computing a ``qop="auth-int"`` digest.

    Returns ``None`` when the body can't be read without consuming it (for
    example a generator, or a file that can't seek back).
    """
    if body is None:
        return b""
    if isinstance(body, (bytes, bytearray)):
        return bytes(body)
    if isinstance(body, str):
        # Match how the body is encoded on the wire: urllib3 2.x uses UTF-8,
        # urllib3 1.x hands str bodies to http.client, which uses Latin-1.
        try:
            return body.encode("iso-8859-1" if is_urllib3_1 else "utf-8")
        except UnicodeEncodeError:
            return None
    read = getattr(body, "read", None)
    tell = getattr(body, "tell", None)
    seek = getattr(body, "seek", None)
    if read is None or tell is None or seek is None:
        return None
    try:
        start = tell()
        data = read()
        seek(start)
    except (OSError, ValueError):
        return None
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data if isinstance(data, bytes) else None


class HTTPDigestAuth(AuthBase):
    """Attaches HTTP Digest Authentication to the given Request object.

    Supports the ``auth`` and ``auth-int`` quality-of-protection (qop)
    modes. ``auth-int`` also protects the request body, so it needs a body
    that can be read without consuming it (bytes, a string or a seekable
    file). When the server offers both modes, ``auth`` is used unless
    ``prefer_auth_int=True``.
    """

    username: bytes | str
    password: bytes | str
    prefer_auth_int: bool
    _thread_local: threading.local
    last_nonce: str
    nonce_count: int
    chal: dict[str, str]
    pos: int | None
    num_401_calls: int | None

    @overload
    def __init__(
        self, username: str, password: str, *, prefer_auth_int: bool = ...
    ) -> None: ...
    @overload
    def __init__(
        self, username: bytes, password: bytes, *, prefer_auth_int: bool = ...
    ) -> None: ...

    def __init__(
        self,
        username: bytes | str,
        password: bytes | str,
        *,
        prefer_auth_int: bool = False,
    ) -> None:
        self.username = username
        self.password = password
        self.prefer_auth_int = prefer_auth_int
        # Keep state in per-thread local storage
        self._thread_local = threading.local()

    def init_per_thread_state(self) -> None:
        # Ensure state is initialized just once per-thread
        if not hasattr(self._thread_local, "init"):
            self._thread_local.init = True
            self._thread_local.last_nonce = ""
            self._thread_local.nonce_count = 0
            self._thread_local.chal = {}
            self._thread_local.pos = None
            self._thread_local.num_401_calls = None

    def _choose_qop(self, qop: str | None, body: Any) -> str | None:
        """Pick the qop mode to answer the challenge with.

        Returns ``""`` for a challenge without qop (RFC 2069 compatibility),
        ``"auth"`` or ``"auth-int"``, or ``None`` if no offered mode can be used.
        """
        if not qop:
            return ""
        offered = {option.strip().lower() for option in qop.split(",")}
        can_hash_body = _body_bytes_for_digest(body) is not None
        if "auth-int" in offered and can_hash_body:
            if self.prefer_auth_int or "auth" not in offered:
                return "auth-int"
        if "auth" in offered:
            return "auth"
        return None

    def build_digest_header(
        self, method: str, url: str, body: Any = None
    ) -> str | None:
        """Build the ``Authorization`` header answering the stored challenge.

        :param body: the request body; only needed for ``qop="auth-int"``.
        :rtype: str
        """

        realm = self._thread_local.chal["realm"]
        nonce = self._thread_local.chal["nonce"]
        qop = self._thread_local.chal.get("qop")
        algorithm = self._thread_local.chal.get("algorithm")
        opaque = self._thread_local.chal.get("opaque")
        hash_utf8 = None

        if algorithm is None:
            _algorithm = "MD5"
        else:
            _algorithm = algorithm.upper()
        # lambdas assume digest modules are imported at the top level
        if _algorithm == "MD5" or _algorithm == "MD5-SESS":

            def md5_utf8(x: str | bytes) -> str:
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.md5(x, usedforsecurity=False).hexdigest()

            hash_utf8 = md5_utf8
        elif _algorithm == "SHA":

            def sha_utf8(x: str | bytes) -> str:
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha1(x, usedforsecurity=False).hexdigest()

            hash_utf8 = sha_utf8
        elif _algorithm == "SHA-256":

            def sha256_utf8(x: str | bytes) -> str:
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha256(x, usedforsecurity=False).hexdigest()

            hash_utf8 = sha256_utf8
        elif _algorithm == "SHA-512":

            def sha512_utf8(x: str | bytes) -> str:
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha512(x, usedforsecurity=False).hexdigest()

            hash_utf8 = sha512_utf8

        if hash_utf8 is None:
            return None

        def KD(s: str, d: str) -> str:
            return hash_utf8(f"{s}:{d}")

        chosen_qop = self._choose_qop(qop, body)
        if chosen_qop is None:
            return None

        p_parsed = urlparse(url)
        #: path is request-uri defined in RFC 2616 which should not be empty
        path = p_parsed.path or "/"
        if p_parsed.query:
            path += f"?{p_parsed.query}"

        A1 = f"{self.username}:{realm}:{self.password}"
        # For auth-int, A2 also covers a hash of the request body (RFC 7616).
        body_hash = ""
        if chosen_qop == "auth-int":
            body_hash = ":" + hash_utf8(cast(bytes, _body_bytes_for_digest(body)))
        A2 = f"{method}:{path}{body_hash}"

        HA1 = hash_utf8(A1)
        HA2 = hash_utf8(A2)

        if nonce == self._thread_local.last_nonce:
            self._thread_local.nonce_count += 1
        else:
            self._thread_local.nonce_count = 1
        ncvalue = f"{self._thread_local.nonce_count:08x}"
        s = str(self._thread_local.nonce_count).encode("utf-8")
        s += nonce.encode("utf-8")
        s += time.ctime().encode("utf-8")
        s += os.urandom(8)

        cnonce = hashlib.sha1(s, usedforsecurity=False).hexdigest()[:16]
        if _algorithm == "MD5-SESS":
            HA1 = hash_utf8(f"{HA1}:{nonce}:{cnonce}")  # type: ignore[reportConstantRedefinition]  # RFC 2617 terminology

        if not chosen_qop:
            respdig = KD(HA1, f"{nonce}:{HA2}")
        else:
            noncebit = f"{nonce}:{ncvalue}:{cnonce}:{chosen_qop}:{HA2}"
            respdig = KD(HA1, noncebit)

        self._thread_local.last_nonce = nonce

        # XXX should the partial digests be encoded too?
        base = (
            f'username="{self.username}", realm="{realm}", nonce="{nonce}", '
            f'uri="{path}", response="{respdig}"'
        )
        if opaque:
            base += f', opaque="{opaque}"'
        if algorithm:
            base += f', algorithm="{algorithm}"'
        if chosen_qop:
            base += f', qop="{chosen_qop}", nc={ncvalue}, cnonce="{cnonce}"'

        return f"Digest {base}"

    def handle_redirect(self, r: Response, **kwargs: Any) -> None:
        """Reset num_401_calls counter on redirects."""
        if r.is_redirect:
            self._thread_local.num_401_calls = 1

    def handle_401(self, r: Response, **kwargs: Any) -> Response:
        """
        Takes the given response and tries digest-auth, if needed.

        :rtype: requests.Response
        """

        # If response is not 4xx, do not auth
        # See https://github.com/psf/requests/issues/3772
        if not 400 <= r.status_code < 500:
            self._thread_local.num_401_calls = 1
            return r

        if self._thread_local.pos is not None:
            # Rewind the file position indicator of the body to where
            # it was to resend the request.
            if (seek := getattr(r.request.body, "seek", None)) is not None:
                seek(self._thread_local.pos)
        s_auth = r.headers.get("www-authenticate", "")

        if "digest" in s_auth.lower() and self._thread_local.num_401_calls < 2:
            self._thread_local.num_401_calls += 1
            pat = re.compile(r"digest ", flags=re.IGNORECASE)
            self._thread_local.chal = parse_dict_header(pat.sub("", s_auth, count=1))

            # Consume content and release the original connection
            # to allow our new request to reuse the same one.
            r.content
            r.close()
            prep = r.request.copy()
            cookie_jar = cast("CookieJar", prep._cookies)
            extract_cookies_to_jar(cookie_jar, r.request, r.raw)
            prep.prepare_cookies(cookie_jar)

            _digest_auth = self.build_digest_header(
                cast(str, prep.method), cast(str, prep.url), prep.body
            )
            if _digest_auth:
                prep.headers["Authorization"] = _digest_auth
            _r = r.connection.send(prep, **kwargs)
            _r.history.append(r)
            _r.request = prep

            return _r

        self._thread_local.num_401_calls = 1
        return r

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        # Initialize per-thread state, if needed
        self.init_per_thread_state()
        # If we have a saved nonce, skip the 401
        if self._thread_local.last_nonce:
            _digest_auth = self.build_digest_header(
                cast(str, r.method), cast(str, r.url), r.body
            )
            if _digest_auth:
                r.headers["Authorization"] = _digest_auth
        if (tell := getattr(r.body, "tell", None)) is not None:
            self._thread_local.pos = tell()
        else:
            # In the case of HTTPDigestAuth being reused and the body of
            # the previous request was a file-like object, pos has the
            # file position of the previous body. Ensure it's set to
            # None.
            self._thread_local.pos = None
        r.register_hook("response", self.handle_401)
        r.register_hook("response", self.handle_redirect)
        self._thread_local.num_401_calls = 1

        return r

    def __eq__(self, other: object) -> bool:
        return all(
            [
                self.username == getattr(other, "username", None),
                self.password == getattr(other, "password", None),
            ]
        )

    def __ne__(self, other: Any) -> bool:
        return not self == other
