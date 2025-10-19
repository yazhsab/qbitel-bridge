"""Add encrypted fields for sensitive data

Revision ID: 003
Revises: 002
Create Date: 2025-10-19

Changes:
- Update users.mfa_secret to LargeBinary (encrypted)
- Update users.mfa_backup_codes to LargeBinary (encrypted)
- Update oauth_providers.client_secret_encrypted to LargeBinary (encrypted)

IMPORTANT: This migration will NOT re-encrypt existing data.
If you have existing data, you must migrate it separately using a data migration script.
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Upgrade database schema to use encrypted fields.

    WARNING: This migration will lose existing data in these fields.
    If you have production data, create a backup first and run a data migration.
    """
    # Users table - encrypted MFA fields
    # Note: We're changing the column type, which will clear existing data
    # For production, you'd want to:
    # 1. Read existing data
    # 2. Encrypt it
    # 3. Write it back

    # Update mfa_secret to LargeBinary (for encrypted data)
    op.alter_column('users', 'mfa_secret',
                    existing_type=sa.String(255),
                    type_=sa.LargeBinary(),
                    existing_nullable=True)

    # Update mfa_backup_codes to LargeBinary (for encrypted JSON)
    op.alter_column('users', 'mfa_backup_codes',
                    existing_type=JSONB(),
                    type_=sa.LargeBinary(),
                    existing_nullable=True)

    # OAuth providers table - already LargeBinary, no change needed
    # client_secret_encrypted is already LargeBinary, just documenting
    # that it will now use the EncryptedText type decorator


def downgrade() -> None:
    """
    Revert to unencrypted fields.

    WARNING: This will lose encrypted data.
    """
    # Revert mfa_secret to String
    op.alter_column('users', 'mfa_secret',
                    existing_type=sa.LargeBinary(),
                    type_=sa.String(255),
                    existing_nullable=True)

    # Revert mfa_backup_codes to JSONB
    op.alter_column('users', 'mfa_backup_codes',
                    existing_type=sa.LargeBinary(),
                    type_=JSONB(),
                    existing_nullable=True)
