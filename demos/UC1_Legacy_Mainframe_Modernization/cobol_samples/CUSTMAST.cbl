       IDENTIFICATION DIVISION.
       PROGRAM-ID. CUSTMAST.
       AUTHOR. QBITEL-BRIDGE-DEMO.
       DATE-WRITTEN. 1985-03-15.
      *================================================================*
      * CUSTOMER MASTER FILE MAINTENANCE PROGRAM                       *
      * THIS PROGRAM MANAGES THE CUSTOMER MASTER FILE FOR THE          *
      * MAINFRAME BANKING SYSTEM. IT HAS BEEN IN PRODUCTION SINCE 1985.*
      *================================================================*

       ENVIRONMENT DIVISION.
       CONFIGURATION SECTION.
       SOURCE-COMPUTER. IBM-390.
       OBJECT-COMPUTER. IBM-390.

       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT CUSTOMER-FILE ASSIGN TO CUSTMAST
               ORGANIZATION IS INDEXED
               ACCESS MODE IS DYNAMIC
               RECORD KEY IS CUST-ID
               FILE STATUS IS WS-FILE-STATUS.
           SELECT TRANSACTION-FILE ASSIGN TO TRANFILE
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-TRAN-STATUS.
           SELECT REPORT-FILE ASSIGN TO CUSTRPT
               ORGANIZATION IS SEQUENTIAL.

       DATA DIVISION.
       FILE SECTION.

       FD  CUSTOMER-FILE.
       01  CUSTOMER-RECORD.
           05  CUST-ID                 PIC 9(10).
           05  CUST-NAME.
               10  CUST-FIRST-NAME     PIC X(20).
               10  CUST-LAST-NAME      PIC X(30).
           05  CUST-ADDRESS.
               10  CUST-STREET         PIC X(40).
               10  CUST-CITY           PIC X(25).
               10  CUST-STATE          PIC X(2).
               10  CUST-ZIP            PIC 9(5).
           05  CUST-PHONE              PIC 9(10).
           05  CUST-SSN                PIC 9(9).
           05  CUST-DOB                PIC 9(8).
           05  CUST-ACCT-TYPE          PIC X(1).
               88  CHECKING            VALUE 'C'.
               88  SAVINGS             VALUE 'S'.
               88  MONEY-MARKET        VALUE 'M'.
           05  CUST-BALANCE            PIC S9(11)V99 COMP-3.
           05  CUST-CREDIT-LIMIT       PIC S9(9)V99 COMP-3.
           05  CUST-OPEN-DATE          PIC 9(8).
           05  CUST-LAST-ACTIVITY      PIC 9(8).
           05  CUST-STATUS             PIC X(1).
               88  ACTIVE              VALUE 'A'.
               88  INACTIVE            VALUE 'I'.
               88  CLOSED              VALUE 'C'.
           05  FILLER                  PIC X(20).

       FD  TRANSACTION-FILE.
       01  TRANSACTION-RECORD.
           05  TRAN-ID                 PIC 9(12).
           05  TRAN-CUST-ID            PIC 9(10).
           05  TRAN-TYPE               PIC X(2).
               88  DEPOSIT             VALUE 'DP'.
               88  WITHDRAWAL          VALUE 'WD'.
               88  TRANSFER            VALUE 'TF'.
               88  INQUIRY             VALUE 'IQ'.
           05  TRAN-AMOUNT             PIC S9(9)V99 COMP-3.
           05  TRAN-DATE               PIC 9(8).
           05  TRAN-TIME               PIC 9(6).
           05  TRAN-STATUS             PIC X(1).
           05  FILLER                  PIC X(30).

       FD  REPORT-FILE.
       01  REPORT-LINE                 PIC X(132).

       WORKING-STORAGE SECTION.
       01  WS-FILE-STATUS              PIC X(2).
           88  WS-FILE-OK              VALUE '00'.
           88  WS-FILE-EOF             VALUE '10'.
           88  WS-FILE-NOT-FOUND       VALUE '23'.
       01  WS-TRAN-STATUS              PIC X(2).
       01  WS-PROCESS-FLAG             PIC X(1).
           88  WS-CONTINUE             VALUE 'Y'.
           88  WS-STOP                 VALUE 'N'.
       01  WS-ERROR-COUNT              PIC 9(5) VALUE 0.
       01  WS-RECORD-COUNT             PIC 9(7) VALUE 0.
       01  WS-CURRENT-DATE             PIC 9(8).
       01  WS-CURRENT-TIME             PIC 9(6).

       01  WS-DISPLAY-BALANCE          PIC Z(10)9.99-.
       01  WS-DISPLAY-AMOUNT           PIC Z(8)9.99-.

       01  WS-REPORT-HEADER.
           05  FILLER                  PIC X(50) VALUE SPACES.
           05  FILLER                  PIC X(32) VALUE
               'CUSTOMER MASTER MAINTENANCE RPT'.
           05  FILLER                  PIC X(50) VALUE SPACES.

       01  WS-REPORT-DETAIL.
           05  WS-RPT-CUST-ID          PIC 9(10).
           05  FILLER                  PIC X(2) VALUE SPACES.
           05  WS-RPT-CUST-NAME        PIC X(50).
           05  FILLER                  PIC X(2) VALUE SPACES.
           05  WS-RPT-BALANCE          PIC Z(10)9.99-.
           05  FILLER                  PIC X(2) VALUE SPACES.
           05  WS-RPT-STATUS           PIC X(10).

       PROCEDURE DIVISION.

       0000-MAIN-PROCESS.
           PERFORM 1000-INITIALIZE
           PERFORM 2000-PROCESS-TRANSACTIONS
               UNTIL WS-STOP
           PERFORM 9000-FINALIZE
           STOP RUN.

       1000-INITIALIZE.
           MOVE 'Y' TO WS-PROCESS-FLAG
           OPEN I-O CUSTOMER-FILE
           IF NOT WS-FILE-OK
               DISPLAY 'ERROR OPENING CUSTOMER FILE: ' WS-FILE-STATUS
               MOVE 'N' TO WS-PROCESS-FLAG
           END-IF
           OPEN INPUT TRANSACTION-FILE
           IF WS-TRAN-STATUS NOT = '00'
               DISPLAY 'ERROR OPENING TRANSACTION FILE'
               MOVE 'N' TO WS-PROCESS-FLAG
           END-IF
           OPEN OUTPUT REPORT-FILE
           ACCEPT WS-CURRENT-DATE FROM DATE YYYYMMDD
           ACCEPT WS-CURRENT-TIME FROM TIME
           WRITE REPORT-LINE FROM WS-REPORT-HEADER.

       2000-PROCESS-TRANSACTIONS.
           READ TRANSACTION-FILE INTO TRANSACTION-RECORD
               AT END
                   MOVE 'N' TO WS-PROCESS-FLAG
               NOT AT END
                   PERFORM 3000-PROCESS-SINGLE-TRAN
           END-READ.

       3000-PROCESS-SINGLE-TRAN.
           ADD 1 TO WS-RECORD-COUNT
           MOVE TRAN-CUST-ID TO CUST-ID
           READ CUSTOMER-FILE
           IF WS-FILE-OK
               EVALUATE TRUE
                   WHEN DEPOSIT
                       PERFORM 4000-PROCESS-DEPOSIT
                   WHEN WITHDRAWAL
                       PERFORM 5000-PROCESS-WITHDRAWAL
                   WHEN TRANSFER
                       PERFORM 6000-PROCESS-TRANSFER
                   WHEN INQUIRY
                       PERFORM 7000-PROCESS-INQUIRY
                   WHEN OTHER
                       ADD 1 TO WS-ERROR-COUNT
               END-EVALUATE
           ELSE
               ADD 1 TO WS-ERROR-COUNT
               DISPLAY 'CUSTOMER NOT FOUND: ' TRAN-CUST-ID
           END-IF.

       4000-PROCESS-DEPOSIT.
           ADD TRAN-AMOUNT TO CUST-BALANCE
           MOVE WS-CURRENT-DATE TO CUST-LAST-ACTIVITY
           REWRITE CUSTOMER-RECORD
           PERFORM 8000-WRITE-REPORT.

       5000-PROCESS-WITHDRAWAL.
           IF TRAN-AMOUNT > CUST-BALANCE
               DISPLAY 'INSUFFICIENT FUNDS FOR CUST: ' CUST-ID
               ADD 1 TO WS-ERROR-COUNT
           ELSE
               SUBTRACT TRAN-AMOUNT FROM CUST-BALANCE
               MOVE WS-CURRENT-DATE TO CUST-LAST-ACTIVITY
               REWRITE CUSTOMER-RECORD
               PERFORM 8000-WRITE-REPORT
           END-IF.

       6000-PROCESS-TRANSFER.
           IF TRAN-AMOUNT > CUST-BALANCE
               DISPLAY 'INSUFFICIENT FUNDS FOR TRANSFER'
               ADD 1 TO WS-ERROR-COUNT
           ELSE
               SUBTRACT TRAN-AMOUNT FROM CUST-BALANCE
               MOVE WS-CURRENT-DATE TO CUST-LAST-ACTIVITY
               REWRITE CUSTOMER-RECORD
               PERFORM 8000-WRITE-REPORT
           END-IF.

       7000-PROCESS-INQUIRY.
           MOVE CUST-BALANCE TO WS-DISPLAY-BALANCE
           DISPLAY 'CUSTOMER: ' CUST-ID ' BALANCE: ' WS-DISPLAY-BALANCE
           PERFORM 8000-WRITE-REPORT.

       8000-WRITE-REPORT.
           MOVE CUST-ID TO WS-RPT-CUST-ID
           STRING CUST-FIRST-NAME DELIMITED BY '  '
                  ' ' DELIMITED BY SIZE
                  CUST-LAST-NAME DELIMITED BY '  '
                  INTO WS-RPT-CUST-NAME
           MOVE CUST-BALANCE TO WS-RPT-BALANCE
           EVALUATE CUST-STATUS
               WHEN 'A'
                   MOVE 'ACTIVE' TO WS-RPT-STATUS
               WHEN 'I'
                   MOVE 'INACTIVE' TO WS-RPT-STATUS
               WHEN 'C'
                   MOVE 'CLOSED' TO WS-RPT-STATUS
           END-EVALUATE
           WRITE REPORT-LINE FROM WS-REPORT-DETAIL.

       9000-FINALIZE.
           DISPLAY 'PROCESSING COMPLETE'
           DISPLAY 'RECORDS PROCESSED: ' WS-RECORD-COUNT
           DISPLAY 'ERRORS ENCOUNTERED: ' WS-ERROR-COUNT
           CLOSE CUSTOMER-FILE
           CLOSE TRANSACTION-FILE
           CLOSE REPORT-FILE.
