       IDENTIFICATION DIVISION.
       PROGRAM-ID. HLTX676.
       AUTHOR. C. MILLER.
       INSTALLATION. QBITEL-BRIDGE.
       DATE-WRITTEN. 2013-06-03.
       DATE-COMPILED. 2026-01-10.
       SECURITY. CONFIDENTIAL.
      *
      * Data archival program for healthcare retention
      * Domain: HEALTHCARE
      * Generated for QBITEL Bridge Training Dataset
      *
       ENVIRONMENT DIVISION.
       CONFIGURATION SECTION.
       SOURCE-COMPUTER. IBM-ZOS.
       OBJECT-COMPUTER. IBM-ZOS.
       SPECIAL-NAMES.
           DECIMAL-POINT IS COMMA.
      *
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT PATIENT-FILE
               ASSIGN TO DDPATIEN
               ORGANIZATION IS INDEXED
               ACCESS MODE IS DYNAMIC
               RECORD KEY IS PATIENT-KEY
               FILE STATUS IS WS-PATI-STATUS.
           SELECT MEDICAL-FILE
               ASSIGN TO DDMEDICA
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-MEDI-STATUS.
       DATA DIVISION.
       FILE SECTION.
       FD PATIENT-FILE
           RECORDING MODE IS F
           BLOCK CONTAINS 0 RECORDS
           RECORD CONTAINS 500 CHARACTERS.
       01 PATIENT-RECORD.
          05 PATIENT-KEY PIC X(10).
          05 PATIENT-PATIENT-ID PIC X(12).
          05 PATIENT-MRN PIC 9(10).
          05 PATIENT-DOS PIC 9(8).
          05 PATIENT-CHARGE PIC S9(7)V99 COMP-3.
          05 FILLER PIC X(20).
       FD MEDICAL-FILE
           RECORDING MODE IS F
           BLOCK CONTAINS 0 RECORDS
           RECORD CONTAINS 80 CHARACTERS.
       01 MEDICAL-RECORD.
          05 MEDICAL-KEY PIC X(10).
          05 MEDICAL-PATIENT-ID PIC X(12).
          05 MEDICAL-MRN PIC 9(10).
          05 MEDICAL-DOS PIC 9(8).
          05 MEDICAL-CHARGE PIC S9(7)V99 COMP-3.
          05 FILLER PIC X(20).
      *
       WORKING-STORAGE SECTION.
      *--- PROGRAM CONSTANTS ---
       01 WS-CONSTANTS.
           05 WS-PROGRAM-NAME      PIC X(8) VALUE 'PATIENT-'.
           05 WS-PROGRAM-VERSION   PIC X(6) VALUE '01.00'.
      *
      *--- FILE STATUS VARIABLES ---
       01 WS-FILE-STATUS-VARS.
           05 WS-PATI-STATUS     PIC XX VALUE SPACES.
           05 WS-MEDI-STATUS     PIC XX VALUE SPACES.
      *
      *--- WORKING VARIABLES ---
       01 WS-WORK-AREAS.
          05 WS-RECORD-TYPE PIC X(1).
          05 WS-FIELD-1 PIC X(20).
          05 WS-AMOUNT PIC S9(11)V99 COMP-3 VALUE 0.
          05 WS-RATE PIC S9(3)V9(4) COMP-3 VALUE 0.
          05 WS-TAX-RATE PIC S9(3)V9(4) COMP-3 VALUE 0.
          05 WS-RESULT PIC S9(13)V99 COMP-3 VALUE 0.
          05 WS-TAX PIC S9(11)V99 COMP-3 VALUE 0.
          05 WS-TOTAL PIC S9(13)V99 COMP-3 VALUE 0.
          05 WS-DATE PIC 9(8) VALUE 0.
          05 WS-STATUS-CODE PIC X(1).
          05 WS-PREV-KEY PIC X(10).
          05 WS-SAVE-AREA PIC X(100).
          05 WS-RETURN-CODE PIC S9(4) COMP VALUE 0.
      *
      *--- COUNTERS AND ACCUMULATORS ---
       01 WS-COUNTERS.
           05 WS-RECORDS-READ      PIC 9(9) COMP VALUE 0.
           05 WS-RECORDS-WRITTEN   PIC 9(9) COMP VALUE 0.
           05 WS-RECORDS-UPDATED   PIC 9(9) COMP VALUE 0.
           05 WS-RECORDS-DELETED   PIC 9(9) COMP VALUE 0.
           05 WS-ERROR-COUNT       PIC 9(9) COMP VALUE 0.
      *
      *--- FLAGS AND SWITCHES ---
       01 WS-FLAGS.
           05 WS-EOF-FLAG          PIC 9 VALUE 0.
              88 END-OF-FILE       VALUE 1.
              88 NOT-END-OF-FILE   VALUE 0.
           05 WS-ERROR-FLAG        PIC 9 VALUE 0.
              88 ERROR-OCCURRED    VALUE 1.
              88 NO-ERROR          VALUE 0.
           05 WS-FIRST-TIME        PIC 9 VALUE 1.
              88 IS-FIRST-TIME     VALUE 1.
              88 NOT-FIRST-TIME    VALUE 0.
      *
      *--- DATE AND TIME ---
       01 WS-DATE-TIME.
           05 WS-CURRENT-DATE.
              10 WS-YEAR           PIC 9(4).
              10 WS-MONTH          PIC 9(2).
              10 WS-DAY            PIC 9(2).
           05 WS-CURRENT-TIME.
              10 WS-HOUR           PIC 9(2).
              10 WS-MINUTE         PIC 9(2).
              10 WS-SECOND         PIC 9(2).
       PROCEDURE DIVISION.
      *===============================================================
      * MAIN PROGRAM LOGIC
      *===============================================================
       0000-MAIN-LOGIC.
           PERFORM 1000-INITIALIZATION
           PERFORM 2000-PROCESS-MAIN UNTIL END-OF-FILE
           PERFORM 9000-TERMINATION
           STOP RUN.
      *
      *===============================================================
      * INITIALIZATION
      *===============================================================
       1000-INITIALIZATION.
           PERFORM 1100-INIT-VARIABLES
           PERFORM 1200-OPEN-FILES
           PERFORM 1300-READ-FIRST-RECORD.
      *
       1100-INIT-VARIABLES.
           INITIALIZE WS-COUNTERS
           INITIALIZE WS-FLAGS
           MOVE FUNCTION CURRENT-DATE TO WS-DATE-TIME.
      *
       1200-OPEN-FILES.
           OPEN INPUT PATIENT-FILE
           IF WS-PATI-STATUS NOT = '00'
               DISPLAY 'ERROR OPENING PATIENT-FILE: ' WS-PATI-STATUS
               SET ERROR-OCCURRED TO TRUE
               PERFORM 9999-ABORT-PROGRAM
           END-IF.
           OPEN INPUT MEDICAL-FILE
           IF WS-MEDI-STATUS NOT = '00'
               DISPLAY 'ERROR OPENING MEDICAL-FILE: ' WS-MEDI-STATUS
               SET ERROR-OCCURRED TO TRUE
               PERFORM 9999-ABORT-PROGRAM
           END-IF.
      *
       1300-READ-FIRST-RECORD.
           PERFORM 2100-READ-RECORD.
      *
      *===============================================================
      * MAIN PROCESSING
      *===============================================================
       2000-PROCESS-MAIN.
           PERFORM 2200-PROCESS-RECORD
           PERFORM 2100-READ-RECORD.
      *
       2100-READ-RECORD.
           READ PATIENT-FILE
               AT END
                   SET END-OF-FILE TO TRUE
               NOT AT END
                   ADD 1 TO WS-RECORDS-READ
           END-READ.
      *
       2200-PROCESS-RECORD.
           PERFORM 3000-APPLY-BUSINESS-LOGIC
           ADD 1 TO WS-RECORDS-WRITTEN.
      *
       3000-APPLY-BUSINESS-LOGIC.
           CONTINUE.
      *
      *===============================================================
      * TERMINATION
      *===============================================================
       9000-TERMINATION.
           PERFORM 9100-CLOSE-FILES
           PERFORM 9200-DISPLAY-STATISTICS.
      *
       9100-CLOSE-FILES.
           CLOSE PATIENT-FILE.
           CLOSE MEDICAL-FILE.
      *
       9200-DISPLAY-STATISTICS.
           DISPLAY '========================================='
           DISPLAY 'PROGRAM STATISTICS'
           DISPLAY '========================================='
           DISPLAY 'RECORDS READ:    ' WS-RECORDS-READ
           DISPLAY 'RECORDS WRITTEN: ' WS-RECORDS-WRITTEN
           DISPLAY 'RECORDS UPDATED: ' WS-RECORDS-UPDATED
           DISPLAY 'ERRORS:          ' WS-ERROR-COUNT
           DISPLAY '========================================='.
      *
       9999-ABORT-PROGRAM.
           DISPLAY 'PROGRAM ABORTED DUE TO ERROR'
           PERFORM 9100-CLOSE-FILES
           STOP RUN.
