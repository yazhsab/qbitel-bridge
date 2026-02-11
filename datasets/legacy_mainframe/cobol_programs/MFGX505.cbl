       IDENTIFICATION DIVISION.
       PROGRAM-ID. MFGX505.
       AUTHOR. S. BROWN.
       INSTALLATION. QBITEL-BRIDGE.
       DATE-WRITTEN. 1998-12-18.
       DATE-COMPILED. 2026-01-10.
       SECURITY. CONFIDENTIAL.
      *
      * Data archival program for manufacturing retention
      * Domain: MANUFACTURING
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
           SELECT MASTER-FILE
               ASSIGN TO DDMASTER
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-MAST-STATUS.
       DATA DIVISION.
       FILE SECTION.
       FD MASTER-FILE
           RECORDING MODE IS F
           BLOCK CONTAINS 0 RECORDS
           RECORD CONTAINS 80 CHARACTERS.
       01 MASTER-RECORD.
          05 MASTER-KEY PIC X(10).
          05 MASTER-ID PIC 9(10).
          05 MASTER-DATA PIC X(50).
          05 MASTER-AMOUNT PIC S9(9)V99 COMP-3.
          05 FILLER PIC X(20).
      *
       WORKING-STORAGE SECTION.
      *--- PROGRAM CONSTANTS ---
       01 WS-CONSTANTS.
           05 WS-PROGRAM-NAME      PIC X(8) VALUE 'MASTER-F'.
           05 WS-PROGRAM-VERSION   PIC X(6) VALUE '01.00'.
      *
      *--- FILE STATUS VARIABLES ---
       01 WS-FILE-STATUS-VARS.
           05 WS-MAST-STATUS     PIC XX VALUE SPACES.
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
           OPEN INPUT MASTER-FILE
           IF WS-MAST-STATUS NOT = '00'
               DISPLAY 'ERROR OPENING MASTER-FILE: ' WS-MAST-STATUS
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
           READ MASTER-FILE
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
           CLOSE MASTER-FILE.
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
