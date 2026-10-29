from __future__ import annotations

import base64
import hashlib
import time
from dataclasses import dataclass

import bcrypt
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


class AuthError(Exception):
    pass


def _prehash(password: str) -> bytes:
    # bcrypt only reads the first 72 bytes. A SHA-256 prehash makes every character count.
    return base64.b64encode(hashlib.sha256(password.encode("utf-8")).digest())


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash(password), hashed.encode("ascii"))
    except ValueError:
        return False


@dataclass(frozen=True)
class Claims:
    user_id: str
    email: str


class TokenManager:
    """ES256 (ECDSA P-256) JWTs. Only the private key can sign; the public key verifies."""

    def __init__(self, private_pem: str | None = None, ttl_seconds: int = 7 * 24 * 3600) -> None:
        if private_pem:
            key = serialization.load_pem_private_key(private_pem.encode(), password=None)
            if not isinstance(key, ec.EllipticCurvePrivateKey):
                raise ValueError("expected an EC private key")
            self._key = key
        else:
            self._key = ec.generate_private_key(ec.SECP256R1())
        self.ttl = ttl_seconds

    @property
    def public_pem(self) -> str:
        return (
            self._key.public_key()
            .public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            )
            .decode()
        )

    def issue(self, user_id: str, email: str, now: float | None = None) -> str:
        iat = int(now if now is not None else time.time())
        payload = {"sub": user_id, "email": email, "iat": iat, "exp": iat + self.ttl}
        return jwt.encode(payload, self._key, algorithm="ES256")

    def verify(self, token: str) -> Claims:
        try:
            data = jwt.decode(
                token,
                self._key.public_key(),
                algorithms=["ES256"],
                options={"require": ["exp", "sub"]},
            )
        except jwt.PyJWTError as e:
            raise AuthError(str(e)) from e
        return Claims(user_id=data["sub"], email=data.get("email", ""))
