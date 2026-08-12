"""OAuth connections: provider tokens bound to a user.

The access token is stored as **Fernet ciphertext** (see
``agentos.core.security``) so the plaintext token never hits the database.
The plaintext is only materialized in memory for the duration of an API
call or agent run.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from agentos.models.base import Base, TimestampMixin


class OAuthConnection(Base, TimestampMixin):
    """A user's OAuth token for an external provider (GitHub in v1)."""

    __tablename__ = "oauth_connections"
    __table_args__ = (
        UniqueConstraint("user_id", "provider", name="uq_oauth_connections_user_provider"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    provider: Mapped[str] = mapped_column(default="github")
    access_token_encrypted: Mapped[str] = mapped_column()  # Fernet ciphertext
    token_type: Mapped[str] = mapped_column(default="Bearer")
    scope: Mapped[str] = mapped_column(default="")  # space-joined scope list
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
