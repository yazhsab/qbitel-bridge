"""Tests for LMS/XMSS stateful hash-based signatures (lms_xmss.py)."""

import pytest

from ai_engine.crypto.lms_xmss import (
    LmsAlgorithm,
    LmotsAlgorithm,
    XmssAlgorithm,
    InMemoryStateBackend,
    StateExhaustedError,
    StateLockError,
    CNSA2Profile,
)


class TestLmsAlgorithm:
    """Tests for LMS algorithm parameter sets."""

    @pytest.mark.parametrize("algo,height,max_sigs", [
        (LmsAlgorithm.LMS_SHA256_M32_H5, 5, 32),
        (LmsAlgorithm.LMS_SHA256_M32_H10, 10, 1024),
        (LmsAlgorithm.LMS_SHA256_M32_H15, 15, 32768),
        (LmsAlgorithm.LMS_SHA256_M32_H20, 20, 1048576),
        (LmsAlgorithm.LMS_SHA256_M32_H25, 25, 33554432),
    ])
    def test_tree_height_and_max_signatures(self, algo, height, max_sigs):
        assert algo.tree_height == height
        assert algo.max_signatures == max_sigs

    @pytest.mark.parametrize("algo", list(LmsAlgorithm))
    def test_hash_properties(self, algo):
        assert algo.hash_function == "sha256"
        assert algo.hash_length == 32

    @pytest.mark.parametrize("algo", list(LmsAlgorithm))
    def test_type_code_is_int(self, algo):
        assert isinstance(algo.type_code, int)
        assert algo.type_code > 0


class TestLmotsAlgorithm:
    """Tests for LM-OTS parameter sets."""

    @pytest.mark.parametrize("algo,w", [
        (LmotsAlgorithm.LMOTS_SHA256_N32_W1, 1),
        (LmotsAlgorithm.LMOTS_SHA256_N32_W2, 2),
        (LmotsAlgorithm.LMOTS_SHA256_N32_W4, 4),
        (LmotsAlgorithm.LMOTS_SHA256_N32_W8, 8),
    ])
    def test_winternitz_parameter(self, algo, w):
        assert algo.winternitz_parameter == w

    @pytest.mark.parametrize("algo", list(LmotsAlgorithm))
    def test_signature_size_positive(self, algo):
        assert algo.signature_size > 0
        # Signature = 4 + n + p * n
        expected = 4 + algo.hash_length + algo.chain_count * algo.hash_length
        assert algo.signature_size == expected

    def test_w1_largest_signature(self):
        """W=1 has most chains → largest signature."""
        w1 = LmotsAlgorithm.LMOTS_SHA256_N32_W1
        w8 = LmotsAlgorithm.LMOTS_SHA256_N32_W8
        assert w1.signature_size > w8.signature_size


class TestXmssAlgorithm:
    """Tests for XMSS algorithm parameter sets."""

    @pytest.mark.parametrize("algo,height", [
        (XmssAlgorithm.XMSS_SHA2_10_256, 10),
        (XmssAlgorithm.XMSS_SHA2_16_256, 16),
        (XmssAlgorithm.XMSS_SHA2_20_256, 20),
        (XmssAlgorithm.XMSS_SHAKE_10_256, 10),
        (XmssAlgorithm.XMSS_SHAKE_16_256, 16),
        (XmssAlgorithm.XMSS_SHAKE_20_256, 20),
    ])
    def test_tree_height(self, algo, height):
        assert algo.tree_height == height
        assert algo.max_signatures == 2 ** height

    def test_sha2_hash_function(self):
        assert XmssAlgorithm.XMSS_SHA2_10_256.hash_function == "sha256"

    def test_shake_hash_function(self):
        assert XmssAlgorithm.XMSS_SHAKE_10_256.hash_function == "shake256"

    @pytest.mark.parametrize("algo", list(XmssAlgorithm))
    def test_hash_length(self, algo):
        assert algo.hash_length == 32


class TestInMemoryStateBackend:
    """Tests for InMemoryStateBackend."""

    def test_initial_state_is_none(self):
        backend = InMemoryStateBackend()
        assert backend.load_state("test-key") is None

    def test_save_and_load(self):
        backend = InMemoryStateBackend()
        backend.save_state("key-1", 42)
        assert backend.load_state("key-1") == 42

    def test_increment_state(self):
        backend = InMemoryStateBackend()
        backend.save_state("key-1", 0)
        backend.save_state("key-1", 1)
        backend.save_state("key-1", 2)
        assert backend.load_state("key-1") == 2

    def test_independent_keys(self):
        backend = InMemoryStateBackend()
        backend.save_state("key-a", 10)
        backend.save_state("key-b", 20)
        assert backend.load_state("key-a") == 10
        assert backend.load_state("key-b") == 20


class TestExceptions:
    """Tests for stateful signature exceptions."""

    def test_state_exhausted_error(self):
        err = StateExhaustedError("key-1", "LMS-SHA256-H5", 32)
        assert "key-1" in str(err)
        assert err.key_id == "key-1"
        assert err.max_signatures == 32

    def test_state_lock_error(self):
        err = StateLockError("key-2")
        assert "key-2" in str(err)


class TestCNSA2Profile:
    """Tests for CNSA 2.0 compliance profiles."""

    def test_profile_members_exist(self):
        # CNSA2Profile should have firmware, software, and CA profiles
        members = list(CNSA2Profile)
        assert len(members) >= 1

    @pytest.mark.parametrize("profile", list(CNSA2Profile))
    def test_profiles_have_properties(self, profile):
        # Each profile should have usable string value
        assert isinstance(profile.value, str)
        assert len(profile.value) > 0
