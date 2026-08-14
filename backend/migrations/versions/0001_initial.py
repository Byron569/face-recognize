"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-02-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── cameras ─────────────────────────────────────────
    op.create_table(
        "cameras",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False, server_default=""),
        sa.Column("source", sa.String(512), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False, server_default="640"),
        sa.Column("height", sa.Integer(), nullable=False, server_default="480"),
        sa.Column("profile", sa.String(32), nullable=False, server_default="desktop"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("config", JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # ── identities ──────────────────────────────────────
    op.create_table(
        "identities",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("avatar_path", sa.String(512), nullable=True),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_identities_name", "identities", ["name"])

    # ── identity_embeddings ─────────────────────────────
    op.create_table(
        "identity_embeddings",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "identity_id",
            sa.Uuid(),
            sa.ForeignKey("identities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("embedding", ARRAY(sa.Float()), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("source", sa.String(32), nullable=False, server_default="image"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_identity_embeddings_identity_id", "identity_embeddings", ["identity_id"])

    # ── recognition_logs ────────────────────────────────
    op.create_table(
        "recognition_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("camera_id", sa.String(64), nullable=False),
        sa.Column(
            "identity_id",
            sa.Uuid(),
            sa.ForeignKey("identities.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("similarity", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_recognition_logs_camera_id", "recognition_logs", ["camera_id"])
    op.create_index("ix_recognition_logs_created_at", "recognition_logs", ["created_at"])

    # ── events(扩展事件预留)─────────────────────────────
    # enum 由 create_table 自动创建(create_type=True),无需手动 create
    event_type = sa.Enum(
        "recognition",
        "fall_detected",
        "fall_potential",
        "fall_recovered",
        "intrusion",
        "loitering",
        name="event_type",
        create_type=True,
    )

    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("event_type", event_type, nullable=False),
        sa.Column("camera_id", sa.String(64), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=True),
        sa.Column(
            "identity_id",
            sa.Uuid(),
            sa.ForeignKey("identities.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("payload", JSONB(), nullable=False, server_default="{}"),
        sa.Column("snapshot_path", sa.String(512), nullable=True),
        sa.Column("acknowledged", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_events_event_type", "events", ["event_type"])
    op.create_index("ix_events_camera_id", "events", ["camera_id"])
    op.create_index("ix_events_created_at", "events", ["created_at"])
    op.create_index("ix_events_acknowledged", "events", ["acknowledged"])


def downgrade() -> None:
    op.drop_table("events")
    op.drop_table("recognition_logs")
    op.drop_table("identity_embeddings")
    op.drop_table("identities")
    op.drop_table("cameras")
    sa.Enum(name="event_type").drop(op.get_bind(), checkfirst=True)
