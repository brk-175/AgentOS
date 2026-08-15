"""Security primitive tests: Fernet token cipher + signed session tokens."""

import uuid

import pytest

from agentos.core.security import SessionTokens, TokenCipher, TokenCipherError

SECRET = "test-secret-key-0123456789"


@pytest.fixture()
def cipher() -> TokenCipher:
    return TokenCipher(fernet_key="k3g9h9vBdFrZCJbGwAmEAbC6r2zHg1dZcE4sFj8mU0U=")


@pytest.fixture()
def sessions() -> SessionTokens:
    return SessionTokens(SECRET)


def test_cipher_round_trip(cipher: TokenCipher) -> None:
    plaintext = "gho_1234567890abcdef"
    encrypted = cipher.encrypt_token(plaintext)
    assert encrypted != plaintext
    assert cipher.decrypt_token(encrypted) == plaintext


def test_cipher_does_not_share_keys(cipher: TokenCipher) -> None:
    other = TokenCipher(fernet_key="Lk9sHf2VgNpQwErTyUaZdCvBxMn0jKl8o1Ik2Uj3HnG=")
    encrypted = other.encrypt_token("secret")
    with pytest.raises(TokenCipherError):
        cipher.decrypt_token(encrypted)


def test_cipher_rejects_tampered_ciphertext(cipher: TokenCipher) -> None:
    encrypted = cipher.encrypt_token("gho_secret")
    tampered = encrypted[:-4] + ("AAAA" if not encrypted.endswith("AAAA") else "BBBB")
    with pytest.raises(TokenCipherError):
        cipher.decrypt_token(tampered)


def test_cipher_requires_valid_key() -> None:
    with pytest.raises(ValueError):
        TokenCipher("not-a-fernert-key")
    with pytest.raises(ValueError):
        TokenCipher("")


def test_session_round_trip(sessions: SessionTokens) -> None:
    user_id = uuid.uuid4()
    token = sessions.create(user_id)
    assert sessions.verify(token) == user_id


def test_session_rejects_tampering(sessions: SessionTokens) -> None:
    user_id = uuid.uuid4()
    token = sessions.create(user_id)
    mid = len(token) // 2
    replacement = "A" if token[mid] != "A" else "B"
    tampered = token[:mid] + replacement + token[mid + 1 :]
    assert sessions.verify(tampered) is None


def test_session_rejects_expired(sessions: SessionTokens) -> None:
    token = sessions.create(uuid.uuid4(), ttl_seconds=-60)
    assert sessions.verify(token) is None


def test_session_rejects_garbage(sessions: SessionTokens) -> None:
    assert sessions.verify("") is None
    assert sessions.verify("not-a-token") is None
    assert sessions.verify("a" * 5000) is None


def test_session_requires_strong_secret() -> None:
    with pytest.raises(ValueError):
        SessionTokens("short")
