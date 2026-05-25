"""Tests for the unified PQC engine (pqc_unified.py)."""

import pytest

from ai_engine.crypto.pqc_unified import (
    PQCAlgorithm,
    PQCSecurityLevel,
    DomainProfile,
    PQCEngine,
    create_enterprise_engine,
    create_healthcare_engine,
    create_automotive_engine,
    create_cnsa2_engine,
)


class TestPQCAlgorithm:
    """Tests for PQCAlgorithm enum properties."""

    @pytest.mark.parametrize("algo", [
        PQCAlgorithm.MLKEM_512,
        PQCAlgorithm.MLKEM_768,
        PQCAlgorithm.MLKEM_1024,
    ])
    def test_kem_algorithms_are_kem(self, algo):
        assert algo.is_kem is True
        assert algo.is_signature is False

    @pytest.mark.parametrize("algo", [
        PQCAlgorithm.MLDSA_44,
        PQCAlgorithm.DILITHIUM_3,
        PQCAlgorithm.FALCON_512,
    ])
    def test_signature_algorithms_are_signature(self, algo):
        assert algo.is_signature is True
        assert algo.is_kem is False

    @pytest.mark.parametrize("algo", [
        PQCAlgorithm.LMS_SHA256_H20,
        PQCAlgorithm.LMS_SHA256_H25,
        PQCAlgorithm.XMSS_SHA2_20,
    ])
    def test_stateful_algorithms(self, algo):
        assert algo.is_stateful is True

    def test_non_stateful_algorithms(self):
        assert PQCAlgorithm.MLKEM_768.is_stateful is False
        assert PQCAlgorithm.DILITHIUM_3.is_stateful is False

    @pytest.mark.parametrize("algo", [
        PQCAlgorithm.MLKEM_1024,
        PQCAlgorithm.MLDSA_87,
        PQCAlgorithm.LMS_SHA256_H20,
        PQCAlgorithm.LMS_SHA256_H25,
        PQCAlgorithm.XMSS_SHA2_20,
    ])
    def test_cnsa2_approved(self, algo):
        assert algo.is_cnsa2_approved is True

    def test_non_cnsa2_approved(self):
        assert PQCAlgorithm.MLKEM_512.is_cnsa2_approved is False
        assert PQCAlgorithm.FALCON_512.is_cnsa2_approved is False

    def test_nist_levels(self):
        assert PQCAlgorithm.MLKEM_512.nist_level == 1
        assert PQCAlgorithm.MLKEM_768.nist_level == 3
        assert PQCAlgorithm.MLKEM_1024.nist_level == 5
        assert PQCAlgorithm.MLDSA_44.nist_level == 2
        assert PQCAlgorithm.FALCON_1024.nist_level == 5


class TestPQCSecurityLevel:
    """Tests for PQCSecurityLevel enum."""

    def test_all_levels_exist(self):
        assert PQCSecurityLevel.LEVEL_1.value == 1
        assert PQCSecurityLevel.LEVEL_2.value == 2
        assert PQCSecurityLevel.LEVEL_3.value == 3
        assert PQCSecurityLevel.LEVEL_5.value == 5


class TestDomainProfile:
    """Tests for DomainProfile enum and its algorithm defaults."""

    def test_all_profiles_exist(self):
        profiles = [
            DomainProfile.ENTERPRISE,
            DomainProfile.HEALTHCARE,
            DomainProfile.AUTOMOTIVE,
            DomainProfile.AVIATION,
            DomainProfile.INDUSTRIAL,
            DomainProfile.TELECOM,
            DomainProfile.GOVERNMENT,
            DomainProfile.CNSA2_DEFENSE,
        ]
        assert len(profiles) == 8

    @pytest.mark.parametrize("profile", list(DomainProfile))
    def test_default_kem_defined(self, profile):
        kem = profile.default_kem
        assert isinstance(kem, PQCAlgorithm)
        assert kem.is_kem

    @pytest.mark.parametrize("profile", list(DomainProfile))
    def test_default_signature_defined(self, profile):
        sig = profile.default_signature
        assert isinstance(sig, PQCAlgorithm)
        assert sig.is_signature

    def test_enterprise_defaults(self):
        assert DomainProfile.ENTERPRISE.default_kem == PQCAlgorithm.MLKEM_768
        assert DomainProfile.ENTERPRISE.default_signature == PQCAlgorithm.DILITHIUM_3

    def test_healthcare_defaults(self):
        assert DomainProfile.HEALTHCARE.default_kem == PQCAlgorithm.MLKEM_512
        assert DomainProfile.HEALTHCARE.default_signature == PQCAlgorithm.FALCON_512

    def test_cnsa2_defaults(self):
        assert DomainProfile.CNSA2_DEFENSE.default_kem == PQCAlgorithm.MLKEM_1024
        assert DomainProfile.CNSA2_DEFENSE.default_signature == PQCAlgorithm.MLDSA_87

    def test_automotive_uses_falcon(self):
        assert DomainProfile.AUTOMOTIVE.default_signature == PQCAlgorithm.FALCON_512


class TestPQCEngine:
    """Tests for PQCEngine initialization and domain profiles."""

    @pytest.mark.parametrize("profile", list(DomainProfile))
    def test_init_all_profiles(self, pqc_fallback_mode, profile):
        engine = PQCEngine(profile)
        assert engine is not None

    def test_engine_has_profile(self, pqc_fallback_mode):
        engine = PQCEngine(DomainProfile.ENTERPRISE)
        assert engine.domain == DomainProfile.ENTERPRISE


class TestFactoryFunctions:
    """Tests for module-level factory functions."""

    def test_create_enterprise_engine(self, pqc_fallback_mode):
        engine = create_enterprise_engine()
        assert engine.domain == DomainProfile.ENTERPRISE

    def test_create_healthcare_engine(self, pqc_fallback_mode):
        engine = create_healthcare_engine()
        assert engine.domain == DomainProfile.HEALTHCARE

    def test_create_automotive_engine(self, pqc_fallback_mode):
        engine = create_automotive_engine()
        assert engine.domain == DomainProfile.AUTOMOTIVE

    def test_create_cnsa2_engine(self, pqc_fallback_mode):
        engine = create_cnsa2_engine()
        assert engine.domain == DomainProfile.CNSA2_DEFENSE
