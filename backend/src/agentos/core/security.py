"""Security primitives: Fernet token encryption + HMAC-signed session tokens.

- ``TokenCipher`` encrypts/decrypts OAuth tokens at rest (ciphertext in DB,
  plaintext materialised in memory only for the duration of a call/run).
- ``SessionTokens`` issues stateless, tamper-proof session cookies
  (``urlsafe_b64(user_id:expiry).urlsafe_b64(HMAC-SHA256)``) signed with
  ``SECRET_KEY``, so the API needs no session store in v1.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
import uuid
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from agentos.core.config import get_settings

SESSION_TTL_SECONDS = 7 * 24 * 3600
_HEADER_SEPARATOR = "."
_MAX_TOKEN_LENGTH = 4096


class TokenCipherError(Exception):
    """Raised when a stored token cannot be decrypted (key mismatch/tampering)."""


class TokenCipher:
    """Fernet symmetric encryption for sensitive values stored at rest."""

    def __init__(self, fernet_key: str) -> None:
        if not fernet_key:
            raise ValueError("FERNET_KEY is required")
        try:
            self._fernet = Fernet(fernet_key.encode())
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "FERNET_KEY is not a valid Fernet key (32 url-safe base64 bytes). "
                'Generate one with: python -c "from cryptography.fernet import '
                'Fernet; print(Fernet.generate_key().decode())"'
            ) from exc

    def encrypt_token(self, plaintext: str) -> str:
        """Encrypt a plaintext token into a Fernet ciphertext string."""
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt_token(self, ciphertext: str) -> str:
        """Decrypt a Fernet ciphertext back to the plaintext token."""
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except InvalidToken as exc:
            raise TokenCipherError("token decryption failed (tampered or key mismatch)") from exc


class SessionTokens:
    """Stateless signed session tokens: ``payload.signature``."""

    def __init__(self, secret_key: str) -> None:
        if len(secret_key) < 16:
            raise ValueError("SECRET_KEY must be at least 16 characters")
        self._secret = secret_key.encode()

    @staticmethod
    def _b64encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode().rstrip("=")

    @staticmethod
    def _b64decode(data: str) -> bytes | None:
        padding = "=" * (-len(data) % 4)
        try:
            return base64.urlsafe_b64decode(data + padding)
        except (ValueError, TypeError):
            return None

    def create(self, user_id: uuid.UUID, ttl_seconds: int = SESSION_TTL_SECONDS) -> str:
        """Issue a signed token valid for ``ttl_seconds``."""
        payload = f"{user_id}:{int(time.time()) + ttl_seconds}".encode()
        body = self._b64encode(payload)
        signature = hmac.new(self._secret, body.encode(), hashlib.sha256).digest()
        return f"{body}{_HEADER_SEPARATOR}{self._b64encode(signature)}"

    def verify(self, token: str) -> uuid.UUID | None:
        """Return the user id if the token is authentic and unexpired, else None."""
        if not token or len(token) > _MAX_TOKEN_LENGTH:
            return None
        parts = token.rsplit(_HEADER_SEPARATOR, 1)
        if len(parts) != 2:
            return None
        body, signature = parts

        expected = hmac.new(self._secret, body.encode(), hashlib.sha256).digest()
        provided = self._b64decode(signature)
        if provided is None or not hmac.compare_digest(expected, provided):
            return None

        payload = self._b64decode(body)
        if payload is None:
            return None
        try:
            user_id, expires = payload.decode().rsplit(":", 1)
            if time.time() > int(expires):
                return None
            return uuid.UUID(user_id)
        except (ValueError, OverflowError):
            return None


@lru_cache
def get_token_cipher() -> TokenCipher:
    """Process-wide ``TokenCipher`` bound to ``FERNET_KEY``."""
    return TokenCipher(get_settings().fernet_key)


@lru_cache
def get_session_tokens() -> SessionTokens:
    """Process-wide ``SessionTokens`` bound to ``SECRET_KEY``."""
    return SessionTokens(get_settings().secret_key)
