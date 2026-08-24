"""reliable event outbox

Revision ID: 0002_event_outbox
Revises: 0001_initial
Create Date: 2026-08-24
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0002_event_outbox"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── events:可靠事件幂等字段 ─────────────────────────
    op.add_column("events", sa.Column("incident_id", sa.String(64), nullable=True))
    op.add_column("events", sa.Column("source_event_id", sa.String(64), nullable=True))
    op.add_column("events", sa.Column("dedupe_key", sa.String(160), nullable=True))
    op.add_column("events", sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "events",
        sa.Column("delivery_mode", sa.String(16), nullable=False, server_default="alert"),
    )
    op.create_index("ix_events_incident_id", "events", ["incident_id"])
    op.create_index("ix_events_source_event_id", "events", ["source_event_id"], unique=True)
    op.create_index("ix_events_dedupe_key", "events", ["dedupe_key"], unique=True)

    # ── event_outbox:通用事务型 outbox ──────────────────
    op.create_table(
        "event_outbox",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column(
            "event_row_id",
            sa.BigInteger(),
            sa.ForeignKey("events.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("dedupe_key", sa.String(160), nullable=False),
        sa.Column("payload", JSONB(), nullable=False, server_default="{}"),
        sa.Column("delivery_mode", sa.String(16), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_unique_constraint("uq_event_outbox_dedupe_key", "event_outbox", ["dedupe_key"])
    op.create_index("ix_event_outbox_delivered_at", "event_outbox", ["delivered_at"])
    op.create_index("ix_event_outbox_next_attempt_at", "event_outbox", ["next_attempt_at"])


def downgrade() -> None:
    op.drop_index("ix_event_outbox_next_attempt_at", table_name="event_outbox")
    op.drop_index("ix_event_outbox_delivered_at", table_name="event_outbox")
    op.drop_constraint("uq_event_outbox_dedupe_key", "event_outbox", type_="unique")
    op.drop_table("event_outbox")

    op.drop_index("ix_events_dedupe_key", table_name="events")
    op.drop_index("ix_events_source_event_id", table_name="events")
    op.drop_index("ix_events_incident_id", table_name="events")
    op.drop_column("events", "delivery_mode")
    op.drop_column("events", "occurred_at")
    op.drop_column("events", "dedupe_key")
    op.drop_column("events", "source_event_id")
    op.drop_column("events", "incident_id")