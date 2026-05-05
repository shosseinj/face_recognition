"""delete created_at from DetectionLog

Revision ID: ed7c39b40c17
Revises: e0b2ebd3fc7e
Create Date: 2026-05-05 05:15:47.471121

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ed7c39b40c17'
down_revision: Union[str, Sequence[str], None] = 'e0b2ebd3fc7e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Use batch mode for SQLite compatibility
    with op.batch_alter_table('DetectionLogs') as batch_op:
        # For SQLite, we need to check if the column already has NOT NULL constraint
        # If not, we can try to alter it (batch mode will recreate table if needed)
        batch_op.alter_column('access_granted',
                           existing_type=sa.BOOLEAN(),
                           nullable=False)
        batch_op.drop_column('created_at')


def downgrade() -> None:
    """Downgrade schema."""
    # Use batch mode for SQLite compatibility
    with op.batch_alter_table('DetectionLogs') as batch_op:
        batch_op.add_column(sa.Column('created_at', sa.DATETIME(), nullable=True))
        batch_op.alter_column('access_granted',
                           existing_type=sa.BOOLEAN(),
                           nullable=True)