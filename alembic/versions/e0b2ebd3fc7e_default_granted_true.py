"""default granted True

Revision ID: e0b2ebd3fc7e
Revises: 8309582e5107
Create Date: 2026-05-04 08:06:58.321475

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e0b2ebd3fc7e'
down_revision: Union[str, Sequence[str], None] = '8309582e5107'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# alembic/versions/e0b2ebd3fc7e_default_granted_true.py

def upgrade():
    # SQLite workaround: recreate table with new schema
    with op.batch_alter_table('DetectionLogs') as batch_op:
        # For SQLite, just update existing rows
        op.execute("UPDATE DetectionLogs SET access_granted = 1 WHERE access_granted IS NULL")

def downgrade():
    with op.batch_alter_table('DetectionLogs') as batch_op:
        # Revert to nullable (SQLite can't remove NOT NULL easily)
        op.execute("UPDATE DetectionLogs SET access_granted = NULL")