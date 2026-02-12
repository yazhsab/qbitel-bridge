"""
PIN Verification Values

Implements PVV (PIN Verification Value) and CVV (Card Verification Value)
generation and verification.
"""

from typing import Optional, Tuple

from ai_engine.domains.banking.security.hsm import (
    HSMProvider,
    HSMKeyHandle,
)


class PVV:
    """
    PIN Verification Value (PVV) operations.

    PVV is a 4-digit value used to verify PINs without storing the PIN.

    Algorithms:
    - IBM 3624: Original IBM algorithm
    - VISA PVV: Used by Visa and most issuers
    """

    def __init__(self, hsm_provider: Optional[HSMProvider] = None):
        """
        Initialize PVV operations.

        Args:
            hsm_provider: HSM for cryptographic operations
        """
        self._hsm = hsm_provider

    def generate_ibm_3624(
        self,
        pin: str,
        pan: str,
        pvk: HSMKeyHandle,
        decimalization_table: str = "0123456789012345",
        validation_data: Optional[str] = None,
    ) -> str:
        """
        Generate IBM 3624 PIN Offset.

        This generates an offset value that can be used with
        the natural PIN to verify the customer PIN.

        Args:
            pin: Customer's PIN
            pan: Primary Account Number
            pvk: PIN Verification Key
            decimalization_table: 16-character table for hex->decimal
            validation_data: Optional validation data

        Returns:
            PIN offset value
        """
        if not self._hsm:
            # Software implementation for development
            return self._generate_ibm_3624_software(pin, pan, decimalization_table)

        # HSM implementation
        return self._hsm.generate_pin_offset(
            pin=pin,
            pan=pan,
            pvk=pvk,
            decimalization_table=decimalization_table,
            validation_data=validation_data,
        )

    def _generate_ibm_3624_software(
        self,
        pin: str,
        pan: str,
        decimalization_table: str,
    ) -> str:
        """Software implementation of IBM 3624 (development only)."""
        # Get validation data from PAN
        clean_pan = pan.replace(" ", "").replace("-", "")
        validation_data = clean_pan[-13:-1]  # 12 rightmost excluding check digit

        # This is a simplified implementation
        # Real implementation would use 3DES encryption with PVK

        # Calculate natural PIN (first 4 digits of encrypted validation data)
        # For demo, just use a simple derivation
        import hashlib

        hash_input = f"{validation_data}:{pin}"
        hash_result = hashlib.sha256(hash_input.encode()).hexdigest()

        # Decimalize the result
        natural_pin = ""
        for c in hash_result[:16]:
            idx = int(c, 16)
            natural_pin += decimalization_table[idx]
            if len(natural_pin) == 4:
                break

        # Calculate offset: (PIN - Natural PIN) mod 10 for each digit
        offset = ""
        for i in range(len(pin)):
            diff = (int(pin[i]) - int(natural_pin[i])) % 10
            offset += str(diff)

        return offset

    def verify_ibm_3624(
        self,
        encrypted_pin_block: bytes,
        pan: str,
        offset: str,
        pvk: HSMKeyHandle,
        zpk: HSMKeyHandle,
        decimalization_table: str = "0123456789012345",
    ) -> bool:
        """
        Verify PIN using IBM 3624 offset.

        Args:
            encrypted_pin_block: Encrypted PIN block
            pan: Primary Account Number
            offset: PIN offset value
            pvk: PIN Verification Key
            zpk: Zone PIN Key (encrypting the PIN block)
            decimalization_table: Decimalization table

        Returns:
            True if PIN is valid
        """
        if not self._hsm:
            raise ValueError("HSM required for PIN verification")

        # Use HSM's verify command
        return self._hsm.verify_pin_ibm3624(
            encrypted_pin_block=encrypted_pin_block,
            pan=pan,
            offset=offset,
            pvk=pvk,
            zpk=zpk,
            decimalization_table=decimalization_table,
        )

    def generate_visa_pvv(
        self,
        pin: str,
        pan: str,
        pvk: HSMKeyHandle,
        pvki: int = 0,
    ) -> str:
        """
        Generate VISA PIN Verification Value.

        Format:
        1. Build 16-digit string: 11 rightmost PAN digits + PVKI + PIN
        2. Encrypt with PVK (3DES)
        3. Extract 4 decimal digits using decimalization

        Args:
            pin: Customer's PIN (4 digits)
            pan: Primary Account Number
            pvk: PIN Verification Key pair
            pvki: PIN Verification Key Index (0-6)

        Returns:
            4-digit PVV
        """
        if len(pin) != 4:
            raise ValueError("PIN must be 4 digits for VISA PVV")

        if not 0 <= pvki <= 6:
            raise ValueError("PVKI must be 0-6")

        if not self._hsm:
            # Software implementation for development
            return self._generate_visa_pvv_software(pin, pan, pvki)

        # HSM implementation
        return self._hsm.generate_visa_pvv(
            pin=pin,
            pan=pan,
            pvk=pvk,
            pvki=pvki,
        )

    def _generate_visa_pvv_software(
        self,
        pin: str,
        pan: str,
        pvki: int,
    ) -> str:
        """Software implementation of VISA PVV (development only)."""
        clean_pan = pan.replace(" ", "").replace("-", "")

        # Build transform data: 11 rightmost digits of PAN + PVKI + PIN
        pan_digits = clean_pan[-12:-1]  # 11 digits (excluding check)
        transform = f"{pan_digits}{pvki}{pin}"

        # For development, use simplified derivation
        import hashlib

        hash_result = hashlib.sha256(transform.encode()).hexdigest()

        # Decimalize: scan for decimal digits
        pvv = ""
        for c in hash_result:
            if c.isdigit() and len(pvv) < 4:
                pvv += c
            elif c in "abcdef" and len(pvv) < 4:
                # Convert A-F to 0-5
                pvv += str(ord(c) - ord("a"))

        return pvv[:4]

    def verify_visa_pvv(
        self,
        encrypted_pin_block: bytes,
        pan: str,
        pvv: str,
        pvk: HSMKeyHandle,
        zpk: HSMKeyHandle,
        pvki: int = 0,
    ) -> bool:
        """
        Verify PIN using VISA PVV.

        Args:
            encrypted_pin_block: Encrypted PIN block
            pan: Primary Account Number
            pvv: Expected PVV
            pvk: PIN Verification Key
            zpk: Zone PIN Key
            pvki: PIN Verification Key Index

        Returns:
            True if PIN is valid
        """
        if not self._hsm:
            raise ValueError("HSM required for PIN verification")

        # Use HSM's verify command
        return self._hsm.verify_visa_pvv(
            encrypted_pin_block=encrypted_pin_block,
            pan=pan,
            pvv=pvv,
            pvk=pvk,
            zpk=zpk,
            pvki=pvki,
        )


class CVV:
    """
    Card Verification Value (CVV/CVC/CVV2) operations.

    CVV variants:
    - CVV1: Encoded in magnetic stripe Track 2
    - CVV2/CVC2: Printed on card (3 digits)
    - iCVV: Used in chip transactions
    """

    def __init__(self, hsm_provider: Optional[HSMProvider] = None):
        """
        Initialize CVV operations.

        Args:
            hsm_provider: HSM for cryptographic operations
        """
        self._hsm = hsm_provider

    def generate_cvv(
        self,
        pan: str,
        expiry: str,  # YYMM format
        service_code: str,  # 3 digits
        cvk_a: HSMKeyHandle,
        cvk_b: HSMKeyHandle,
    ) -> str:
        """
        Generate CVV1.

        Algorithm:
        1. Concatenate PAN + Expiry + Service Code (pad to 32 hex)
        2. Encrypt left 16 with CVK-A
        3. XOR with right 16
        4. Encrypt result with CVK-B
        5. Decimalize to 3 digits

        Args:
            pan: Primary Account Number
            expiry: Expiry date YYMM
            service_code: Service code (3 digits)
            cvk_a: CVK part A
            cvk_b: CVK part B

        Returns:
            3-digit CVV
        """
        if not self._hsm:
            return self._generate_cvv_software(pan, expiry, service_code)

        # HSM implementation
        return self._hsm.generate_cvv(
            pan=pan,
            expiry=expiry,
            service_code=service_code,
            cvk_a=cvk_a,
            cvk_b=cvk_b,
        )

    def _generate_cvv_software(
        self,
        pan: str,
        expiry: str,
        service_code: str,
    ) -> str:
        """Software CVV generation (development only)."""
        clean_pan = pan.replace(" ", "").replace("-", "")

        # Build CVV data
        cvv_data = f"{clean_pan}{expiry}{service_code}"
        cvv_data = cvv_data.ljust(32, "0")

        # Simplified derivation for development
        import hashlib

        hash_input = cvv_data
        hash_result = hashlib.sha256(hash_input.encode()).hexdigest()

        # Decimalize to 3 digits
        cvv = ""
        for c in hash_result:
            if c.isdigit() and len(cvv) < 3:
                cvv += c
            elif c in "abcdef" and len(cvv) < 3:
                cvv += str(ord(c) - ord("a"))

        return cvv[:3]

    def generate_cvv2(
        self,
        pan: str,
        expiry: str,  # MMYY format for CVV2
        cvk_a: HSMKeyHandle,
        cvk_b: HSMKeyHandle,
    ) -> str:
        """
        Generate CVV2 (printed on card back).

        Similar to CVV1 but:
        - Uses expiry in MMYY format
        - Service code replaced with "000"
        - For Visa/MC, generates 3 digits
        - For Amex, generates 4 digits (CID)

        Args:
            pan: Primary Account Number
            expiry: Expiry date MMYY
            cvk_a: CVK part A
            cvk_b: CVK part B

        Returns:
            CVV2 value
        """
        # CVV2 uses "000" as service code
        if not self._hsm:
            return self._generate_cvv_software(pan, expiry, "000")

        # HSM implementation
        return self._hsm.generate_cvv(
            pan=pan,
            expiry=expiry,
            service_code="000",
            cvk_a=cvk_a,
            cvk_b=cvk_b,
        )

    def generate_icvv(
        self,
        pan: str,
        expiry: str,
        cvk_a: HSMKeyHandle,
        cvk_b: HSMKeyHandle,
    ) -> str:
        """
        Generate iCVV for chip transactions.

        Same algorithm as CVV1 but:
        - Service code is "999"
        - Indicates chip present

        Args:
            pan: Primary Account Number
            expiry: Expiry date YYMM
            cvk_a: CVK part A
            cvk_b: CVK part B

        Returns:
            3-digit iCVV
        """
        if not self._hsm:
            return self._generate_cvv_software(pan, expiry, "999")

        # HSM implementation - iCVV uses service code "999"
        return self._hsm.generate_cvv(
            pan=pan,
            expiry=expiry,
            service_code="999",
            cvk_a=cvk_a,
            cvk_b=cvk_b,
        )

    def verify_cvv(
        self,
        pan: str,
        expiry: str,
        service_code: str,
        cvv: str,
        cvk_a: HSMKeyHandle,
        cvk_b: HSMKeyHandle,
    ) -> bool:
        """
        Verify CVV value.

        Args:
            pan: Primary Account Number
            expiry: Expiry date
            service_code: Service code
            cvv: CVV to verify
            cvk_a: CVK part A
            cvk_b: CVK part B

        Returns:
            True if CVV is valid
        """
        if not self._hsm:
            expected = self._generate_cvv_software(pan, expiry, service_code)
            return expected == cvv

        # HSM implementation - generate and compare
        expected = self._hsm.generate_cvv(
            pan=pan,
            expiry=expiry,
            service_code=service_code,
            cvk_a=cvk_a,
            cvk_b=cvk_b,
        )
        # Use constant-time comparison for security
        import secrets

        return secrets.compare_digest(expected, cvv)

    def verify_cvv2(
        self,
        pan: str,
        expiry: str,
        cvv2: str,
        cvk_a: HSMKeyHandle,
        cvk_b: HSMKeyHandle,
    ) -> bool:
        """
        Verify CVV2 value.

        Args:
            pan: Primary Account Number
            expiry: Expiry date MMYY
            cvv2: CVV2 to verify
            cvk_a: CVK part A
            cvk_b: CVK part B

        Returns:
            True if CVV2 is valid
        """
        return self.verify_cvv(pan, expiry, "000", cvv2, cvk_a, cvk_b)
