"""
Unit tests for COBOL parsing functionality.
"""

import pytest
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


class TestBasicCOBOLParser:
    """Tests for basic COBOL parsing (fallback mode)."""

    def test_extract_program_id_standard(self):
        """Test extracting standard PROGRAM-ID."""
        from production_app import _basic_cobol_parse

        source = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. TESTPROG.
       AUTHOR. DEVELOPER.
        """
        result = _basic_cobol_parse(source)
        assert result["program_name"] == "TESTPROG"

    def test_extract_program_id_with_period(self):
        """Test extracting PROGRAM-ID with trailing period."""
        from production_app import _basic_cobol_parse

        source = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. MYPROG.
        """
        result = _basic_cobol_parse(source)
        assert result["program_name"] == "MYPROG"

    def test_extract_program_id_hyphenated(self):
        """Test extracting hyphenated PROGRAM-ID."""
        from production_app import _basic_cobol_parse

        source = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. MY-COBOL-PROG.
        """
        result = _basic_cobol_parse(source)
        assert result["program_name"] == "MY-COBOL-PROG"

    def test_identify_all_divisions(self):
        """Test identifying all four COBOL divisions."""
        from production_app import _basic_cobol_parse

        source = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. TESTPROG.

       ENVIRONMENT DIVISION.
       CONFIGURATION SECTION.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-VAR PIC X(10).

       PROCEDURE DIVISION.
       MAIN-PARA.
           DISPLAY "Hello".
           STOP RUN.
        """
        result = _basic_cobol_parse(source)
        divisions = result["divisions"]

        assert "IDENTIFICATION" in divisions
        assert "ENVIRONMENT" in divisions
        assert "DATA" in divisions
        assert "PROCEDURE" in divisions

    def test_extract_data_structures(self):
        """Test extracting data structure definitions."""
        from production_app import _basic_cobol_parse

        source = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. TESTPROG.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 CUSTOMER-RECORD.
           05 CUST-ID       PIC X(10).
           05 CUST-NAME     PIC X(50).
           05 CUST-BALANCE  PIC 9(9)V99.
       01 WS-EOF            PIC 9.

       PROCEDURE DIVISION.
       MAIN-PARA.
           STOP RUN.
        """
        result = _basic_cobol_parse(source)
        data_structures = result["data_structures"]

        # Should find level 01 and 05 items
        level_01_items = [d for d in data_structures if d["level"] == "01"]
        level_05_items = [d for d in data_structures if d["level"] == "05"]

        assert len(level_01_items) >= 1
        assert len(level_05_items) >= 3

    def test_extract_procedures(self):
        """Test extracting procedure names."""
        from production_app import _basic_cobol_parse

        source = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. TESTPROG.

       DATA DIVISION.

       PROCEDURE DIVISION.
       MAIN-PROCEDURE.
           PERFORM INIT-PARA
           PERFORM PROCESS-PARA
           PERFORM CLEANUP-PARA
           STOP RUN.

       INIT-PARA.
           DISPLAY "INIT".

       PROCESS-PARA.
           DISPLAY "PROCESS".

       CLEANUP-PARA.
           DISPLAY "CLEANUP".
        """
        result = _basic_cobol_parse(source)
        procedures = result["procedures"]
        proc_names = [p["name"] for p in procedures]

        assert len(procedures) >= 3

    def test_calculate_lines_of_code(self):
        """Test lines of code calculation."""
        from production_app import _basic_cobol_parse

        source = """Line 1
Line 2
Line 3
Line 4
Line 5"""
        result = _basic_cobol_parse(source)
        assert result["lines_of_code"] == 5

    def test_complexity_score_simple_program(self):
        """Test complexity score for simple program."""
        from production_app import _basic_cobol_parse

        source = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. SIMPLE.
       DATA DIVISION.
       PROCEDURE DIVISION.
       MAIN.
           STOP RUN.
        """
        result = _basic_cobol_parse(source)
        assert 0 <= result["complexity_score"] <= 1.0

    def test_complexity_score_complex_program(self, sample_cobol_source):
        """Test complexity score increases with program complexity."""
        from production_app import _basic_cobol_parse

        result = _basic_cobol_parse(sample_cobol_source)
        # Complex program should have higher complexity
        assert result["complexity_score"] > 0.1

    def test_modernization_recommendations(self):
        """Test modernization recommendations are generated."""
        from production_app import _basic_cobol_parse

        source = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. LEGACY.
       PROCEDURE DIVISION.
       MAIN.
           STOP RUN.
        """
        result = _basic_cobol_parse(source)
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0

    def test_handles_empty_source(self):
        """Test handling of empty source code."""
        from production_app import _basic_cobol_parse

        result = _basic_cobol_parse("")
        assert result["program_name"] == "UNKNOWN"
        assert result["lines_of_code"] == 1  # Empty string splits to one empty string

    def test_handles_comments(self):
        """Test handling of COBOL comments."""
        from production_app import _basic_cobol_parse

        source = """
      * This is a comment
       IDENTIFICATION DIVISION.
      * Another comment
       PROGRAM-ID. COMMENTED.
       PROCEDURE DIVISION.
       MAIN.
           STOP RUN.
        """
        result = _basic_cobol_parse(source)
        assert result["program_name"] == "COMMENTED"


class TestCOBOLDataTypeDetection:
    """Tests for COBOL data type detection."""

    def test_detect_numeric_field(self):
        """Test detection of numeric PIC clause."""
        from production_app import _basic_cobol_parse

        source = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. TYPES.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-AMOUNT  PIC 9(9)V99.
       PROCEDURE DIVISION.
       MAIN. STOP RUN.
        """
        result = _basic_cobol_parse(source)
        data_structures = result["data_structures"]

        # Should find the numeric field
        assert any("WS-AMOUNT" in str(d) for d in data_structures)

    def test_detect_alphanumeric_field(self):
        """Test detection of alphanumeric PIC clause."""
        from production_app import _basic_cobol_parse

        source = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. TYPES.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-NAME    PIC X(50).
       PROCEDURE DIVISION.
       MAIN. STOP RUN.
        """
        result = _basic_cobol_parse(source)
        data_structures = result["data_structures"]

        assert any("WS-NAME" in str(d) for d in data_structures)


class TestCOBOLFileHandling:
    """Tests for COBOL file handling detection."""

    def test_detect_file_section(self):
        """Test detection of FILE SECTION."""
        from production_app import _basic_cobol_parse

        source = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. FILES.
       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT CUST-FILE ASSIGN TO 'CUSTMAST.DAT'.
       DATA DIVISION.
       FILE SECTION.
       FD CUST-FILE.
       01 CUST-RECORD PIC X(100).
       PROCEDURE DIVISION.
       MAIN.
           STOP RUN.
        """
        result = _basic_cobol_parse(source)
        # Should detect ENVIRONMENT division
        assert "ENVIRONMENT" in result["divisions"]

    def test_detect_indexed_file(self):
        """Test detection of indexed file organization."""
        from production_app import _basic_cobol_parse

        source = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. INDEXED.
       ENVIRONMENT DIVISION.
       FILE-CONTROL.
           SELECT MASTER-FILE ASSIGN TO 'MASTER.DAT'
               ORGANIZATION IS INDEXED
               ACCESS MODE IS DYNAMIC
               RECORD KEY IS REC-KEY.
       DATA DIVISION.
       PROCEDURE DIVISION.
       MAIN. STOP RUN.
        """
        result = _basic_cobol_parse(source)
        # Should parse without error
        assert result["program_name"] == "INDEXED"
