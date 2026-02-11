       IDENTIFICATION DIVISION.
       PROGRAM-ID. ACCTPROC.
       AUTHOR. QBITEL-BRIDGE-DEMO.
       DATE-WRITTEN. 1988-07-22.
      *================================================================*
      * ACCOUNT PROCESSING BATCH PROGRAM                               *
      * PROCESSES NIGHTLY BATCH OF ACCOUNT TRANSACTIONS                *
      * CALCULATES INTEREST, FEES, AND GENERATES STATEMENTS            *
      *================================================================*

       ENVIRONMENT DIVISION.
       CONFIGURATION SECTION.
       SOURCE-COMPUTER. IBM-390.
       OBJECT-COMPUTER. IBM-390.

       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT ACCOUNT-MASTER ASSIGN TO ACCTMAST
               ORGANIZATION IS INDEXED
               ACCESS MODE IS DYNAMIC
               RECORD KEY IS ACCT-NUMBER
               FILE STATUS IS WS-ACCT-STATUS.
           SELECT DAILY-TRANS ASSIGN TO DAYTRANS
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-TRAN-STATUS.
           SELECT INTEREST-RATES ASSIGN TO INTRATES
               ORGANIZATION IS SEQUENTIAL.
           SELECT STATEMENT-FILE ASSIGN TO STMTFILE
               ORGANIZATION IS SEQUENTIAL.
           SELECT ERROR-FILE ASSIGN TO ERRFILE
               ORGANIZATION IS SEQUENTIAL.

       DATA DIVISION.
       FILE SECTION.

       FD  ACCOUNT-MASTER.
       01  ACCOUNT-RECORD.
           05  ACCT-NUMBER             PIC 9(12).
           05  ACCT-BRANCH             PIC 9(4).
           05  ACCT-TYPE               PIC X(2).
               88  CHECKING-ACCT       VALUE 'CK'.
               88  SAVINGS-ACCT        VALUE 'SV'.
               88  CD-ACCT             VALUE 'CD'.
               88  MONEY-MKT           VALUE 'MM'.
               88  LOAN-ACCT           VALUE 'LN'.
           05  ACCT-OWNER-ID           PIC 9(10).
           05  ACCT-BALANCE            PIC S9(13)V99 COMP-3.
           05  ACCT-AVAIL-BAL          PIC S9(13)V99 COMP-3.
           05  ACCT-HOLD-AMT           PIC S9(11)V99 COMP-3.
           05  ACCT-INT-RATE           PIC 9V9(5).
           05  ACCT-INT-ACCRUED        PIC S9(9)V99 COMP-3.
           05  ACCT-LAST-INT-DATE      PIC 9(8).
           05  ACCT-OPEN-DATE          PIC 9(8).
           05  ACCT-MATURITY-DATE      PIC 9(8).
           05  ACCT-MONTHLY-FEE        PIC S9(5)V99 COMP-3.
           05  ACCT-MIN-BALANCE        PIC S9(9)V99 COMP-3.
           05  ACCT-OVERDRAFT-LIMIT    PIC S9(9)V99 COMP-3.
           05  ACCT-YTD-INTEREST       PIC S9(9)V99 COMP-3.
           05  ACCT-YTD-FEES           PIC S9(7)V99 COMP-3.
           05  ACCT-STATUS             PIC X(1).
               88  ACCT-ACTIVE         VALUE 'A'.
               88  ACCT-DORMANT        VALUE 'D'.
               88  ACCT-FROZEN         VALUE 'F'.
               88  ACCT-CLOSED         VALUE 'C'.
           05  ACCT-FLAGS.
               10  ACCT-STMT-FLAG      PIC X(1).
               10  ACCT-OD-FLAG        PIC X(1).
               10  ACCT-HOLD-FLAG      PIC X(1).
               10  ACCT-VIP-FLAG       PIC X(1).
           05  FILLER                  PIC X(40).

       FD  DAILY-TRANS.
       01  TRANS-RECORD.
           05  TR-ACCT-NUMBER          PIC 9(12).
           05  TR-TRAN-CODE            PIC X(3).
               88  TR-DEPOSIT          VALUE 'DEP'.
               88  TR-WITHDRAWAL       VALUE 'WDL'.
               88  TR-CHECK            VALUE 'CHK'.
               88  TR-TRANSFER-OUT     VALUE 'TRO'.
               88  TR-TRANSFER-IN      VALUE 'TRI'.
               88  TR-FEE              VALUE 'FEE'.
               88  TR-INTEREST         VALUE 'INT'.
               88  TR-ADJUSTMENT       VALUE 'ADJ'.
           05  TR-AMOUNT               PIC S9(11)V99 COMP-3.
           05  TR-DATE                 PIC 9(8).
           05  TR-TIME                 PIC 9(6).
           05  TR-REF-NUMBER           PIC X(16).
           05  TR-DESCRIPTION          PIC X(30).
           05  TR-BRANCH               PIC 9(4).
           05  TR-TELLER-ID            PIC X(8).
           05  FILLER                  PIC X(20).

       FD  INTEREST-RATES.
       01  RATE-RECORD.
           05  RATE-ACCT-TYPE          PIC X(2).
           05  RATE-TIER               PIC 9(2).
           05  RATE-MIN-BALANCE        PIC S9(13)V99 COMP-3.
           05  RATE-MAX-BALANCE        PIC S9(13)V99 COMP-3.
           05  RATE-PERCENT            PIC 9V9(5).
           05  RATE-EFF-DATE           PIC 9(8).
           05  FILLER                  PIC X(20).

       FD  STATEMENT-FILE.
       01  STMT-RECORD                 PIC X(200).

       FD  ERROR-FILE.
       01  ERROR-RECORD                PIC X(200).

       WORKING-STORAGE SECTION.
       01  WS-ACCT-STATUS              PIC X(2).
           88  WS-ACCT-OK              VALUE '00'.
           88  WS-ACCT-EOF             VALUE '10'.
           88  WS-ACCT-NOT-FOUND       VALUE '23'.
       01  WS-TRAN-STATUS              PIC X(2).
           88  WS-TRAN-OK              VALUE '00'.
           88  WS-TRAN-EOF             VALUE '10'.

       01  WS-PROCESSING-DATE          PIC 9(8).
       01  WS-PREVIOUS-DATE            PIC 9(8).
       01  WS-PROCESS-FLAG             PIC X(1) VALUE 'Y'.
           88  WS-CONTINUE             VALUE 'Y'.
           88  WS-END-PROCESS          VALUE 'N'.

       01  WS-COUNTERS.
           05  WS-TRANS-READ           PIC 9(9) VALUE 0.
           05  WS-TRANS-APPLIED        PIC 9(9) VALUE 0.
           05  WS-TRANS-REJECTED       PIC 9(9) VALUE 0.
           05  WS-ACCTS-PROCESSED      PIC 9(7) VALUE 0.
           05  WS-INT-CALCULATED       PIC 9(7) VALUE 0.
           05  WS-FEES-ASSESSED        PIC 9(7) VALUE 0.
           05  WS-STMTS-GENERATED      PIC 9(7) VALUE 0.

       01  WS-TOTALS.
           05  WS-TOTAL-DEPOSITS       PIC S9(15)V99 COMP-3 VALUE 0.
           05  WS-TOTAL-WITHDRAWALS    PIC S9(15)V99 COMP-3 VALUE 0.
           05  WS-TOTAL-INTEREST       PIC S9(13)V99 COMP-3 VALUE 0.
           05  WS-TOTAL-FEES           PIC S9(11)V99 COMP-3 VALUE 0.

       01  WS-CALC-FIELDS.
           05  WS-DAILY-RATE           PIC 9V9(8).
           05  WS-DAYS-ELAPSED         PIC 9(3).
           05  WS-INT-AMOUNT           PIC S9(9)V99 COMP-3.
           05  WS-FEE-AMOUNT           PIC S9(7)V99 COMP-3.

       01  WS-ERROR-MSG.
           05  FILLER                  PIC X(10) VALUE 'ERROR:    '.
           05  WS-ERR-ACCT             PIC 9(12).
           05  FILLER                  PIC X(2) VALUE '  '.
           05  WS-ERR-CODE             PIC X(3).
           05  FILLER                  PIC X(2) VALUE '  '.
           05  WS-ERR-DESC             PIC X(60).
           05  FILLER                  PIC X(2) VALUE '  '.
           05  WS-ERR-AMT              PIC Z(10)9.99-.

       01  WS-STMT-HEADER.
           05  FILLER                  PIC X(30) VALUE SPACES.
           05  FILLER                  PIC X(25) VALUE
               'ACCOUNT STATEMENT'.
           05  FILLER                  PIC X(30) VALUE SPACES.

       01  WS-STMT-DETAIL.
           05  WS-SD-DATE              PIC 9999/99/99.
           05  FILLER                  PIC X(2) VALUE SPACES.
           05  WS-SD-DESC              PIC X(30).
           05  FILLER                  PIC X(2) VALUE SPACES.
           05  WS-SD-AMOUNT            PIC Z(10)9.99-.
           05  FILLER                  PIC X(2) VALUE SPACES.
           05  WS-SD-BALANCE           PIC Z(12)9.99-.

       01  WS-RATE-TABLE.
           05  WS-RATE-ENTRY OCCURS 20 TIMES.
               10  WS-RT-TYPE          PIC X(2).
               10  WS-RT-TIER          PIC 9(2).
               10  WS-RT-MIN           PIC S9(13)V99 COMP-3.
               10  WS-RT-MAX           PIC S9(13)V99 COMP-3.
               10  WS-RT-RATE          PIC 9V9(5).
       01  WS-RATE-COUNT               PIC 9(2) VALUE 0.
       01  WS-RATE-IDX                 PIC 9(2).

       PROCEDURE DIVISION.

       0000-MAIN-PROCESS.
           PERFORM 1000-INITIALIZE
           IF WS-CONTINUE
               PERFORM 2000-LOAD-RATES
               PERFORM 3000-PROCESS-TRANSACTIONS
                   UNTIL WS-END-PROCESS
               PERFORM 4000-CALCULATE-INTEREST
               PERFORM 5000-ASSESS-FEES
               PERFORM 6000-GENERATE-STATEMENTS
           END-IF
           PERFORM 9000-FINALIZE
           STOP RUN.

       1000-INITIALIZE.
           ACCEPT WS-PROCESSING-DATE FROM DATE YYYYMMDD
           DISPLAY '*** ACCOUNT PROCESSING STARTED ***'
           DISPLAY 'PROCESSING DATE: ' WS-PROCESSING-DATE

           OPEN I-O ACCOUNT-MASTER
           IF NOT WS-ACCT-OK
               DISPLAY 'ERROR OPENING ACCOUNT MASTER: ' WS-ACCT-STATUS
               MOVE 'N' TO WS-PROCESS-FLAG
           END-IF

           OPEN INPUT DAILY-TRANS
           IF NOT WS-TRAN-OK
               DISPLAY 'ERROR OPENING TRANSACTION FILE'
               MOVE 'N' TO WS-PROCESS-FLAG
           END-IF

           OPEN INPUT INTEREST-RATES
           OPEN OUTPUT STATEMENT-FILE
           OPEN OUTPUT ERROR-FILE.

       2000-LOAD-RATES.
           MOVE 0 TO WS-RATE-COUNT
           PERFORM UNTIL WS-RATE-COUNT >= 20
               READ INTEREST-RATES INTO RATE-RECORD
                   AT END
                       EXIT PERFORM
                   NOT AT END
                       ADD 1 TO WS-RATE-COUNT
                       MOVE RATE-ACCT-TYPE TO
                           WS-RT-TYPE(WS-RATE-COUNT)
                       MOVE RATE-TIER TO
                           WS-RT-TIER(WS-RATE-COUNT)
                       MOVE RATE-MIN-BALANCE TO
                           WS-RT-MIN(WS-RATE-COUNT)
                       MOVE RATE-MAX-BALANCE TO
                           WS-RT-MAX(WS-RATE-COUNT)
                       MOVE RATE-PERCENT TO
                           WS-RT-RATE(WS-RATE-COUNT)
               END-READ
           END-PERFORM
           DISPLAY 'LOADED ' WS-RATE-COUNT ' INTEREST RATE TIERS'.

       3000-PROCESS-TRANSACTIONS.
           READ DAILY-TRANS INTO TRANS-RECORD
               AT END
                   MOVE 'N' TO WS-PROCESS-FLAG
               NOT AT END
                   ADD 1 TO WS-TRANS-READ
                   PERFORM 3100-APPLY-TRANSACTION
           END-READ.

       3100-APPLY-TRANSACTION.
           MOVE TR-ACCT-NUMBER TO ACCT-NUMBER
           READ ACCOUNT-MASTER
           IF NOT WS-ACCT-OK
               PERFORM 3900-LOG-ERROR
           ELSE
               IF ACCT-FROZEN OR ACCT-CLOSED
                   MOVE 'ACCT FROZEN/CLOSED' TO WS-ERR-DESC
                   PERFORM 3900-LOG-ERROR
               ELSE
                   EVALUATE TRUE
                       WHEN TR-DEPOSIT OR TR-TRANSFER-IN
                           PERFORM 3200-PROCESS-CREDIT
                       WHEN TR-WITHDRAWAL OR TR-CHECK OR
                            TR-TRANSFER-OUT
                           PERFORM 3300-PROCESS-DEBIT
                       WHEN TR-FEE
                           PERFORM 3400-PROCESS-FEE
                       WHEN TR-INTEREST
                           PERFORM 3500-PROCESS-INTEREST
                       WHEN TR-ADJUSTMENT
                           PERFORM 3600-PROCESS-ADJUSTMENT
                   END-EVALUATE
               END-IF
           END-IF.

       3200-PROCESS-CREDIT.
           ADD TR-AMOUNT TO ACCT-BALANCE
           ADD TR-AMOUNT TO ACCT-AVAIL-BAL
           ADD TR-AMOUNT TO WS-TOTAL-DEPOSITS
           REWRITE ACCOUNT-RECORD
           ADD 1 TO WS-TRANS-APPLIED.

       3300-PROCESS-DEBIT.
           IF TR-AMOUNT > ACCT-AVAIL-BAL
               IF TR-AMOUNT > (ACCT-AVAIL-BAL +
                              ACCT-OVERDRAFT-LIMIT)
                   MOVE 'INSUFFICIENT FUNDS' TO WS-ERR-DESC
                   PERFORM 3900-LOG-ERROR
               ELSE
                   SUBTRACT TR-AMOUNT FROM ACCT-BALANCE
                   SUBTRACT TR-AMOUNT FROM ACCT-AVAIL-BAL
                   ADD TR-AMOUNT TO WS-TOTAL-WITHDRAWALS
                   MOVE 'Y' TO ACCT-OD-FLAG
                   REWRITE ACCOUNT-RECORD
                   ADD 1 TO WS-TRANS-APPLIED
               END-IF
           ELSE
               SUBTRACT TR-AMOUNT FROM ACCT-BALANCE
               SUBTRACT TR-AMOUNT FROM ACCT-AVAIL-BAL
               ADD TR-AMOUNT TO WS-TOTAL-WITHDRAWALS
               REWRITE ACCOUNT-RECORD
               ADD 1 TO WS-TRANS-APPLIED
           END-IF.

       3400-PROCESS-FEE.
           SUBTRACT TR-AMOUNT FROM ACCT-BALANCE
           SUBTRACT TR-AMOUNT FROM ACCT-AVAIL-BAL
           ADD TR-AMOUNT TO ACCT-YTD-FEES
           ADD TR-AMOUNT TO WS-TOTAL-FEES
           REWRITE ACCOUNT-RECORD
           ADD 1 TO WS-TRANS-APPLIED.

       3500-PROCESS-INTEREST.
           ADD TR-AMOUNT TO ACCT-BALANCE
           ADD TR-AMOUNT TO ACCT-AVAIL-BAL
           ADD TR-AMOUNT TO ACCT-YTD-INTEREST
           ADD TR-AMOUNT TO WS-TOTAL-INTEREST
           REWRITE ACCOUNT-RECORD
           ADD 1 TO WS-TRANS-APPLIED.

       3600-PROCESS-ADJUSTMENT.
           ADD TR-AMOUNT TO ACCT-BALANCE
           ADD TR-AMOUNT TO ACCT-AVAIL-BAL
           REWRITE ACCOUNT-RECORD
           ADD 1 TO WS-TRANS-APPLIED.

       3900-LOG-ERROR.
           ADD 1 TO WS-TRANS-REJECTED
           MOVE TR-ACCT-NUMBER TO WS-ERR-ACCT
           MOVE TR-TRAN-CODE TO WS-ERR-CODE
           MOVE TR-AMOUNT TO WS-ERR-AMT
           WRITE ERROR-RECORD FROM WS-ERROR-MSG.

       4000-CALCULATE-INTEREST.
           DISPLAY 'CALCULATING INTEREST...'
           MOVE LOW-VALUES TO ACCT-NUMBER
           START ACCOUNT-MASTER KEY > ACCT-NUMBER
           IF WS-ACCT-OK
               PERFORM 4100-CALC-INT-LOOP
                   UNTIL WS-ACCT-EOF
           END-IF.

       4100-CALC-INT-LOOP.
           READ ACCOUNT-MASTER NEXT
               AT END
                   SET WS-ACCT-EOF TO TRUE
               NOT AT END
                   IF ACCT-ACTIVE AND
                      (SAVINGS-ACCT OR CD-ACCT OR MONEY-MKT)
                       PERFORM 4200-APPLY-INTEREST
                   END-IF
           END-READ.

       4200-APPLY-INTEREST.
           PERFORM 4300-GET-RATE
           COMPUTE WS-DAILY-RATE =
               WS-RT-RATE(WS-RATE-IDX) / 365
           COMPUTE WS-INT-AMOUNT ROUNDED =
               ACCT-BALANCE * WS-DAILY-RATE
           ADD WS-INT-AMOUNT TO ACCT-INT-ACCRUED
           MOVE WS-PROCESSING-DATE TO ACCT-LAST-INT-DATE
           REWRITE ACCOUNT-RECORD
           ADD 1 TO WS-INT-CALCULATED
           ADD WS-INT-AMOUNT TO WS-TOTAL-INTEREST.

       4300-GET-RATE.
           MOVE 1 TO WS-RATE-IDX
           PERFORM VARYING WS-RATE-IDX FROM 1 BY 1
               UNTIL WS-RATE-IDX > WS-RATE-COUNT
               IF WS-RT-TYPE(WS-RATE-IDX) = ACCT-TYPE
                   AND ACCT-BALANCE >= WS-RT-MIN(WS-RATE-IDX)
                   AND ACCT-BALANCE <= WS-RT-MAX(WS-RATE-IDX)
                   EXIT PERFORM
               END-IF
           END-PERFORM.

       5000-ASSESS-FEES.
           DISPLAY 'ASSESSING MONTHLY FEES...'
           MOVE LOW-VALUES TO ACCT-NUMBER
           START ACCOUNT-MASTER KEY > ACCT-NUMBER
           IF WS-ACCT-OK
               PERFORM 5100-FEE-LOOP
                   UNTIL WS-ACCT-EOF
           END-IF.

       5100-FEE-LOOP.
           READ ACCOUNT-MASTER NEXT
               AT END
                   SET WS-ACCT-EOF TO TRUE
               NOT AT END
                   IF ACCT-ACTIVE AND ACCT-MONTHLY-FEE > 0
                       IF ACCT-BALANCE < ACCT-MIN-BALANCE
                           PERFORM 5200-APPLY-FEE
                       END-IF
                   END-IF
           END-READ.

       5200-APPLY-FEE.
           SUBTRACT ACCT-MONTHLY-FEE FROM ACCT-BALANCE
           SUBTRACT ACCT-MONTHLY-FEE FROM ACCT-AVAIL-BAL
           ADD ACCT-MONTHLY-FEE TO ACCT-YTD-FEES
           ADD ACCT-MONTHLY-FEE TO WS-TOTAL-FEES
           REWRITE ACCOUNT-RECORD
           ADD 1 TO WS-FEES-ASSESSED.

       6000-GENERATE-STATEMENTS.
           DISPLAY 'GENERATING STATEMENTS...'
           MOVE LOW-VALUES TO ACCT-NUMBER
           START ACCOUNT-MASTER KEY > ACCT-NUMBER
           IF WS-ACCT-OK
               PERFORM 6100-STMT-LOOP
                   UNTIL WS-ACCT-EOF
           END-IF.

       6100-STMT-LOOP.
           READ ACCOUNT-MASTER NEXT
               AT END
                   SET WS-ACCT-EOF TO TRUE
               NOT AT END
                   IF ACCT-ACTIVE AND ACCT-STMT-FLAG = 'Y'
                       PERFORM 6200-WRITE-STATEMENT
                   END-IF
           END-READ.

       6200-WRITE-STATEMENT.
           WRITE STMT-RECORD FROM WS-STMT-HEADER
           ADD 1 TO WS-STMTS-GENERATED.

       9000-FINALIZE.
           DISPLAY '*** ACCOUNT PROCESSING COMPLETE ***'
           DISPLAY 'TRANSACTIONS READ:    ' WS-TRANS-READ
           DISPLAY 'TRANSACTIONS APPLIED: ' WS-TRANS-APPLIED
           DISPLAY 'TRANSACTIONS REJECTED:' WS-TRANS-REJECTED
           DISPLAY 'INTEREST CALCULATED:  ' WS-INT-CALCULATED
           DISPLAY 'FEES ASSESSED:        ' WS-FEES-ASSESSED
           DISPLAY 'STATEMENTS GENERATED: ' WS-STMTS-GENERATED

           CLOSE ACCOUNT-MASTER
           CLOSE DAILY-TRANS
           CLOSE INTEREST-RATES
           CLOSE STATEMENT-FILE
           CLOSE ERROR-FILE.
