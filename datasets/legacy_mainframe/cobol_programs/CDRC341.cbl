       IDENTIFICATION DIVISION.
       PROGRAM-ID. CDRC341.
       AUTHOR. M. JOHNSON.
       INSTALLATION. QBITEL-BRIDGE.
       DATE-WRITTEN. 1990-12-11.
       DATE-COMPILED. 2026-01-10.
       SECURITY. CONFIDENTIAL.
      *
      * Calculation engine for telecommunications computations
      * Domain: TELECOMMUNICATIONS
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
           SELECT DETAIL-FILE
               ASSIGN TO DDDETAIL
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-DETA-STATUS.
           SELECT HISTORY-FILE
               ASSIGN TO DDHISTOR
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-HIST-STATUS.
           SELECT REPORT-FILE
               ASSIGN TO DDREPORT
               ORGANIZATION IS INDEXED
               ACCESS MODE IS DYNAMIC
               RECORD KEY IS REPORT-KEY
               FILE STATUS IS WS-REPO-STATUS.
       DATA DIVISION.
       FILE SECTION.
       FD MASTER-FILE
           RECORDING MODE IS F
           BLOCK CONTAINS 0 RECORDS
           RECORD CONTAINS 200 CHARACTERS.
       01 MASTER-RECORD.
          05 MASTER-KEY PIC X(10).
          05 MASTER-ID PIC 9(10).
          05 MASTER-DATA PIC X(50).
          05 MASTER-AMOUNT PIC S9(9)V99 COMP-3.
          05 FILLER PIC X(20).
       FD DETAIL-FILE
           RECORDING MODE IS F
           BLOCK CONTAINS 0 RECORDS
           RECORD CONTAINS 100 CHARACTERS.
       01 DETAIL-RECORD.
          05 DETAIL-KEY PIC X(10).
          05 DETAIL-ID PIC 9(10).
          05 DETAIL-DATA PIC X(50).
          05 DETAIL-AMOUNT PIC S9(9)V99 COMP-3.
          05 FILLER PIC X(20).
       FD HISTORY-FILE
           RECORDING MODE IS F
           BLOCK CONTAINS 0 RECORDS
           RECORD CONTAINS 80 CHARACTERS.
       01 HISTORY-RECORD.
          05 HISTORY-KEY PIC X(10).
          05 HISTORY-ID PIC 9(10).
          05 HISTORY-DATA PIC X(50).
          05 HISTORY-AMOUNT PIC S9(9)V99 COMP-3.
          05 FILLER PIC X(20).
       FD REPORT-FILE
           RECORDING MODE IS F
           BLOCK CONTAINS 0 RECORDS
           RECORD CONTAINS 500 CHARACTERS.
       01 REPORT-RECORD.
          05 REPORT-KEY PIC X(10).
          05 REPORT-ID PIC 9(10).
          05 REPORT-DATA PIC X(50).
          05 REPORT-AMOUNT PIC S9(9)V99 COMP-3.
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
           05 WS-DETA-STATUS     PIC XX VALUE SPACES.
           05 WS-HIST-STATUS     PIC XX VALUE SPACES.
           05 WS-REPO-STATUS     PIC XX VALUE SPACES.
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
           OPEN INPUT MASTER-FILE
           IF WS-MAST-STATUS NOT = '00'
               DISPLAY 'ERROR OPENING MASTER-FILE: ' WS-MAST-STATUS
               SET ERROR-OCCURRED TO TRUE
               PERFORM 9999-ABORT-PROGRAM
           END-IF.
           OPEN I-O DETAIL-FILE
           IF WS-DETA-STATUS NOT = '00'
               DISPLAY 'ERROR OPENING DETAIL-FILE: ' WS-DETA-STATUS
               SET ERROR-OCCURRED TO TRUE
               PERFORM 9999-ABORT-PROGRAM
           END-IF.
           OPEN INPUT HISTORY-FILE
           IF WS-HIST-STATUS NOT = '00'
               DISPLAY 'ERROR OPENING HISTORY-FILE: ' WS-HIST-STATUS
               SET ERROR-OCCURRED TO TRUE
               PERFORM 9999-ABORT-PROGRAM
           END-IF.
           OPEN I-O REPORT-FILE
           IF WS-REPO-STATUS NOT = '00'
               DISPLAY 'ERROR OPENING REPORT-FILE: ' WS-REPO-STATUS
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
           PERFORM 3000-CALCULATE-VALUES
           PERFORM 3100-VALIDATE-RESULTS
           IF NO-ERROR
               PERFORM 3200-STORE-RESULTS
           END-IF.
      *
       3000-CALCULATE-VALUES.
           COMPUTE WS-RESULT = WS-AMOUNT * WS-RATE / 100
           COMPUTE WS-TAX = WS-RESULT * WS-TAX-RATE / 100
           COMPUTE WS-TOTAL = WS-RESULT + WS-TAX.
      *
       3100-VALIDATE-RESULTS.
           SET NO-ERROR TO TRUE
           IF WS-TOTAL < 0
               SET ERROR-OCCURRED TO TRUE
               ADD 1 TO WS-ERROR-COUNT
           END-IF.
      *
       3200-STORE-RESULTS.
           ADD 1 TO WS-RECORDS-WRITTEN.
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
           CLOSE DETAIL-FILE.
           CLOSE HISTORY-FILE.
           CLOSE REPORT-FILE.
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
