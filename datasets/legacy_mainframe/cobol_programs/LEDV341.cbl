       IDENTIFICATION DIVISION.
       PROGRAM-ID. LEDV341.
       AUTHOR. P. WILSON.
       INSTALLATION. QBITEL-BRIDGE.
       DATE-WRITTEN. 1995-01-27.
       DATE-COMPILED. 2026-01-10.
       SECURITY. CONFIDENTIAL.
      *
      * Data validation program for banking input
      * Domain: BANKING
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
           SELECT CUSTOMER-FILE
               ASSIGN TO DDCUSTOM
               ORGANIZATION IS INDEXED
               ACCESS MODE IS DYNAMIC
               RECORD KEY IS CUSTOMER-KEY
               FILE STATUS IS WS-CUST-STATUS.
           SELECT ACCOUNT-FILE
               ASSIGN TO DDACCOUN
               ORGANIZATION IS INDEXED
               ACCESS MODE IS DYNAMIC
               RECORD KEY IS ACCOUNT-KEY
               FILE STATUS IS WS-ACCO-STATUS.
           SELECT TRANSACT-FILE
               ASSIGN TO DDTRANSA
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-TRAN-STATUS.
           SELECT BALANCE-FILE
               ASSIGN TO DDBALANC
               ORGANIZATION IS INDEXED
               ACCESS MODE IS DYNAMIC
               RECORD KEY IS BALANCE-KEY
               FILE STATUS IS WS-BALA-STATUS.
       DATA DIVISION.
       FILE SECTION.
       FD CUSTOMER-FILE
           RECORDING MODE IS F
           BLOCK CONTAINS 0 RECORDS
           RECORD CONTAINS 200 CHARACTERS.
       01 CUSTOMER-RECORD.
          05 CUSTOMER-KEY PIC X(10).
          05 CUSTOMER-ACCOUNT-NO PIC 9(12).
          05 CUSTOMER-AMOUNT PIC S9(11)V99 COMP-3.
          05 CUSTOMER-DATE PIC 9(8).
          05 CUSTOMER-STATUS PIC X(1).
          05 FILLER PIC X(20).
       FD ACCOUNT-FILE
           RECORDING MODE IS F
           BLOCK CONTAINS 0 RECORDS
           RECORD CONTAINS 1000 CHARACTERS.
       01 ACCOUNT-RECORD.
          05 ACCOUNT-KEY PIC X(10).
          05 ACCOUNT-ACCOUNT-NO PIC 9(12).
          05 ACCOUNT-AMOUNT PIC S9(11)V99 COMP-3.
          05 ACCOUNT-DATE PIC 9(8).
          05 ACCOUNT-STATUS PIC X(1).
          05 FILLER PIC X(20).
       FD TRANSACT-FILE
           RECORDING MODE IS F
           BLOCK CONTAINS 0 RECORDS
           RECORD CONTAINS 200 CHARACTERS.
       01 TRANSACT-RECORD.
          05 TRANSACT-KEY PIC X(10).
          05 TRANSACT-ACCOUNT-NO PIC 9(12).
          05 TRANSACT-AMOUNT PIC S9(11)V99 COMP-3.
          05 TRANSACT-DATE PIC 9(8).
          05 TRANSACT-STATUS PIC X(1).
          05 FILLER PIC X(20).
       FD BALANCE-FILE
           RECORDING MODE IS F
           BLOCK CONTAINS 0 RECORDS
           RECORD CONTAINS 100 CHARACTERS.
       01 BALANCE-RECORD.
          05 BALANCE-KEY PIC X(10).
          05 BALANCE-ACCOUNT-NO PIC 9(12).
          05 BALANCE-AMOUNT PIC S9(11)V99 COMP-3.
          05 BALANCE-DATE PIC 9(8).
          05 BALANCE-STATUS PIC X(1).
          05 FILLER PIC X(20).
      *
       WORKING-STORAGE SECTION.
      *--- PROGRAM CONSTANTS ---
       01 WS-CONSTANTS.
           05 WS-PROGRAM-NAME      PIC X(8) VALUE 'CUSTOMER'.
           05 WS-PROGRAM-VERSION   PIC X(6) VALUE '01.00'.
      *
      *--- FILE STATUS VARIABLES ---
       01 WS-FILE-STATUS-VARS.
           05 WS-CUST-STATUS     PIC XX VALUE SPACES.
           05 WS-ACCO-STATUS     PIC XX VALUE SPACES.
           05 WS-TRAN-STATUS     PIC XX VALUE SPACES.
           05 WS-BALA-STATUS     PIC XX VALUE SPACES.
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
           OPEN INPUT CUSTOMER-FILE
           IF WS-CUST-STATUS NOT = '00'
               DISPLAY 'ERROR OPENING CUSTOMER-FILE: ' WS-CUST-STATUS
               SET ERROR-OCCURRED TO TRUE
               PERFORM 9999-ABORT-PROGRAM
           END-IF.
           OPEN I-O ACCOUNT-FILE
           IF WS-ACCO-STATUS NOT = '00'
               DISPLAY 'ERROR OPENING ACCOUNT-FILE: ' WS-ACCO-STATUS
               SET ERROR-OCCURRED TO TRUE
               PERFORM 9999-ABORT-PROGRAM
           END-IF.
           OPEN I-O TRANSACT-FILE
           IF WS-TRAN-STATUS NOT = '00'
               DISPLAY 'ERROR OPENING TRANSACT-FILE: ' WS-TRAN-STATUS
               SET ERROR-OCCURRED TO TRUE
               PERFORM 9999-ABORT-PROGRAM
           END-IF.
           OPEN INPUT BALANCE-FILE
           IF WS-BALA-STATUS NOT = '00'
               DISPLAY 'ERROR OPENING BALANCE-FILE: ' WS-BALA-STATUS
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
           READ CUSTOMER-FILE
               AT END
                   SET END-OF-FILE TO TRUE
               NOT AT END
                   ADD 1 TO WS-RECORDS-READ
           END-READ.
      *
       2200-PROCESS-RECORD.
           PERFORM 3000-VALIDATE-FIELDS
           IF NO-ERROR
               PERFORM 3100-WRITE-VALID-RECORD
           ELSE
               PERFORM 3200-WRITE-ERROR-RECORD
           END-IF.
      *
       3000-VALIDATE-FIELDS.
           SET NO-ERROR TO TRUE
           PERFORM 3010-VALIDATE-NUMERIC-FIELDS
           PERFORM 3020-VALIDATE-DATE-FIELDS
           PERFORM 3030-VALIDATE-CODE-FIELDS.
      *
       3010-VALIDATE-NUMERIC-FIELDS.
           IF WS-AMOUNT NOT NUMERIC
               SET ERROR-OCCURRED TO TRUE
               ADD 1 TO WS-ERROR-COUNT
           END-IF.
      *
       3020-VALIDATE-DATE-FIELDS.
           IF WS-DATE < 19000101 OR WS-DATE > 99991231
               SET ERROR-OCCURRED TO TRUE
               ADD 1 TO WS-ERROR-COUNT
           END-IF.
      *
       3030-VALIDATE-CODE-FIELDS.
           EVALUATE WS-STATUS-CODE
               WHEN 'A' CONTINUE
               WHEN 'I' CONTINUE
               WHEN 'C' CONTINUE
               WHEN OTHER
                   SET ERROR-OCCURRED TO TRUE
                   ADD 1 TO WS-ERROR-COUNT
           END-EVALUATE.
      *
       3100-WRITE-VALID-RECORD.
           ADD 1 TO WS-RECORDS-WRITTEN.
      *
       3200-WRITE-ERROR-RECORD.
           ADD 1 TO WS-ERROR-COUNT.
      *
      *===============================================================
      * TERMINATION
      *===============================================================
       9000-TERMINATION.
           PERFORM 9100-CLOSE-FILES
           PERFORM 9200-DISPLAY-STATISTICS.
      *
       9100-CLOSE-FILES.
           CLOSE CUSTOMER-FILE.
           CLOSE ACCOUNT-FILE.
           CLOSE TRANSACT-FILE.
           CLOSE BALANCE-FILE.
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
