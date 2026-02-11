"""
PIN Block Translator

Translates PIN blocks between formats and encryption keys.
"""

from typing import Optional, Tuple

from ai_engine.domains.banking.security.hsm import (
    HSMProvider,
    HSMKeyHandle,
    HSMAlgorithm,
)
from ai_engine.domains.banking.security.pin_security.pin_block import (
    PINBlock,
    PINBlockFormat,
)


class PINTranslatorError(Exception):
    """Exception for PIN translation errors."""
    pass


class PINTranslator:
    """
    PIN Block Translator.

    Translates PIN blocks:
    - Between different encryption keys (re-encryption)
    - Between different PIN block formats
    - Between clear and encrypted forms

    This is critical for payment networks where PIN blocks
    must be translated as they pass through different zones.
    """

    def __init__(self, hsm_provider: Optional[HSMProvider] = None):
        """
        Initialize PIN translator.

        Args:
            hsm_provider: HSM for cryptographic operations
        """
        self._hsm = hsm_provider

    def encrypt_pin_block(
        self,
        clear_block: PINBlock,
        zpk: HSMKeyHandle,
    ) -> PINBlock:
        """
        Encrypt a clear PIN block.

        Args:
            clear_block: Clear (unencrypted) PIN block
            zpk: Zone PIN Key for encryption

        Returns:
            Encrypted PIN block
        """
        if clear_block.is_encrypted:
            raise PINTranslatorError("PIN block is already encrypted")

        if not self._hsm:
            raise PINTranslatorError("HSM required for encryption")

        # Encrypt based on format
        if clear_block.format == PINBlockFormat.ISO_4:
            # AES encryption for Format 4
            result = self._hsm.encrypt(
                key_handle=zpk,
                plaintext=clear_block.block_data,
                algorithm=HSMAlgorithm.AES_ECB,
            )
        else:
            # 3DES encryption for other formats
            result = self._hsm.encrypt(
                key_handle=zpk,
                plaintext=clear_block.block_data,
                algorithm=HSMAlgorithm.DES3_CBC,
                iv=bytes(8),  # Zero IV for PIN block encryption
            )

        return PINBlock(
            format=clear_block.format,
            block_data=result.ciphertext,
            pan=clear_block.pan,
            is_encrypted=True,
            key_identifier=zpk.key_id,
        )

    def decrypt_pin_block(
        self,
        encrypted_block: PINBlock,
        zpk: HSMKeyHandle,
    ) -> PINBlock:
        """
        Decrypt an encrypted PIN block.

        Args:
            encrypted_block: Encrypted PIN block
            zpk: Zone PIN Key for decryption

        Returns:
            Clear PIN block
        """
        if not encrypted_block.is_encrypted:
            raise PINTranslatorError("PIN block is not encrypted")

        if not self._hsm:
            raise PINTranslatorError("HSM required for decryption")

        # Decrypt based on format
        if encrypted_block.format == PINBlockFormat.ISO_4:
            result = self._hsm.decrypt(
                key_handle=zpk,
                ciphertext=encrypted_block.block_data,
                algorithm=HSMAlgorithm.AES_ECB,
            )
        else:
            result = self._hsm.decrypt(
                key_handle=zpk,
                ciphertext=encrypted_block.block_data,
                algorithm=HSMAlgorithm.DES3_CBC,
                iv=bytes(8),
            )

        return PINBlock(
            format=encrypted_block.format,
            block_data=result.plaintext,
            pan=encrypted_block.pan,
            is_encrypted=False,
        )

    def translate_key(
        self,
        encrypted_block: PINBlock,
        source_zpk: HSMKeyHandle,
        dest_zpk: HSMKeyHandle,
    ) -> PINBlock:
        """
        Translate PIN block from one ZPK to another.

        This re-encrypts the PIN block under a different key
        without exposing the clear PIN.

        Args:
            encrypted_block: PIN block encrypted under source_zpk
            source_zpk: Source Zone PIN Key
            dest_zpk: Destination Zone PIN Key

        Returns:
            PIN block encrypted under dest_zpk
        """
        if not self._hsm:
            raise PINTranslatorError("HSM required for key translation")

        # In a real HSM, this would be done in a single operation
        # without exposing the clear PIN (using HSM's translate command)

        # For software implementation (development only):
        clear_block = self.decrypt_pin_block(encrypted_block, source_zpk)
        return self.encrypt_pin_block(clear_block, dest_zpk)

    def translate_format(
        self,
        pin_block: PINBlock,
        target_format: PINBlockFormat,
        pan: Optional[str] = None,
        zpk: Optional[HSMKeyHandle] = None,
    ) -> PINBlock:
        """
        Translate PIN block to a different format.

        Args:
            pin_block: Source PIN block
            target_format: Target format
            pan: PAN (required for some format translations)
            zpk: ZPK if block is encrypted

        Returns:
            PIN block in target format
        """
        # If encrypted, decrypt first
        if pin_block.is_encrypted:
            if not zpk:
                raise PINTranslatorError("ZPK required for encrypted block translation")
            clear_block = self.decrypt_pin_block(pin_block, zpk)
        else:
            clear_block = pin_block

        # Extract PIN
        pin = clear_block.extract_pin(pan)

        # Use provided PAN or original
        pan_to_use = pan or clear_block.pan

        # Create new block in target format
        if target_format == PINBlockFormat.ISO_0:
            if not pan_to_use:
                raise PINTranslatorError("PAN required for ISO Format 0")
            new_block = PINBlock.create_iso_format_0(pin, pan_to_use)

        elif target_format == PINBlockFormat.ISO_1:
            new_block = PINBlock.create_iso_format_1(pin, 0)

        elif target_format == PINBlockFormat.ISO_2:
            new_block = PINBlock.create_iso_format_2(pin)

        elif target_format == PINBlockFormat.ISO_3:
            if not pan_to_use:
                raise PINTranslatorError("PAN required for ISO Format 3")
            new_block = PINBlock.create_iso_format_3(pin, pan_to_use)

        elif target_format == PINBlockFormat.ISO_4:
            if not pan_to_use:
                raise PINTranslatorError("PAN required for ISO Format 4")
            new_block = PINBlock.create_iso_format_4(pin, pan_to_use)

        else:
            raise PINTranslatorError(f"Translation to {target_format} not supported")

        # Re-encrypt if original was encrypted
        if pin_block.is_encrypted and zpk:
            new_block = self.encrypt_pin_block(new_block, zpk)

        return new_block

    def translate_key_and_format(
        self,
        encrypted_block: PINBlock,
        source_zpk: HSMKeyHandle,
        dest_zpk: HSMKeyHandle,
        target_format: PINBlockFormat,
        pan: Optional[str] = None,
    ) -> PINBlock:
        """
        Translate both key and format in one operation.

        Args:
            encrypted_block: Source encrypted PIN block
            source_zpk: Source Zone PIN Key
            dest_zpk: Destination Zone PIN Key
            target_format: Target format
            pan: PAN if needed for format translation

        Returns:
            PIN block in target format encrypted under dest_zpk
        """
        # Translate format (handles decryption internally)
        translated = self.translate_format(
            pin_block=encrypted_block,
            target_format=target_format,
            pan=pan,
            zpk=source_zpk,
        )

        # The translated block is clear, now encrypt with dest key
        return self.encrypt_pin_block(translated, dest_zpk)

    def verify_pin_blocks_match(
        self,
        block1: PINBlock,
        block2: PINBlock,
        zpk1: Optional[HSMKeyHandle] = None,
        zpk2: Optional[HSMKeyHandle] = None,
        pan: Optional[str] = None,
    ) -> bool:
        """
        Verify that two PIN blocks contain the same PIN.

        Args:
            block1: First PIN block
            block2: Second PIN block
            zpk1: ZPK for block1 if encrypted
            zpk2: ZPK for block2 if encrypted
            pan: PAN for extraction

        Returns:
            True if blocks contain same PIN
        """
        # Decrypt if needed
        if block1.is_encrypted:
            if not zpk1:
                raise PINTranslatorError("ZPK required for encrypted block1")
            block1 = self.decrypt_pin_block(block1, zpk1)

        if block2.is_encrypted:
            if not zpk2:
                raise PINTranslatorError("ZPK required for encrypted block2")
            block2 = self.decrypt_pin_block(block2, zpk2)

        # Extract PINs
        pin1 = block1.extract_pin(pan or block1.pan)
        pin2 = block2.extract_pin(pan or block2.pan)

        return pin1 == pin2
