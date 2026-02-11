      *****************************************************************
      * TRANSACTION RECORD LAYOUT
      * Financial transaction record for the core banking ledger
      *
      * Record Type: Fixed-length 300 bytes
      * Character Set: EBCDIC
      * Used by: BATCH-POSTING, REAL-TIME-AUTH, EOD-PROCESSING
      *****************************************************************
       01  TRANSACTION-RECORD.
           05  TXN-HEADER.
               10  TXN-RECORD-TYPE           PIC X(2).
                   88  TXN-DEBIT             VALUE 'DB'.
                   88  TXN-CREDIT            VALUE 'CR'.
                   88  TXN-REVERSAL          VALUE 'RV'.
                   88  TXN-ADJUSTMENT        VALUE 'AJ'.
               10  TXN-VERSION               PIC 9(2).
               10  TXN-SEQUENCE              PIC 9(12).

           05  TXN-IDENTITY.
               10  TXN-ID                    PIC X(20).
               10  TXN-REF-NUMBER            PIC X(16).
               10  TXN-BATCH-ID              PIC 9(8).
               10  TXN-SOURCE-SYSTEM         PIC X(4).
                   88  TXN-ATM               VALUE 'ATM '.
                   88  TXN-POS               VALUE 'POS '.
                   88  TXN-WIRE              VALUE 'WIRE'.
                   88  TXN-ACH               VALUE 'ACH '.
                   88  TXN-BRANCH            VALUE 'BRCH'.
                   88  TXN-ONLINE            VALUE 'ONLN'.

           05  TXN-ACCOUNT-INFO.
               10  TXN-ACCT-FROM             PIC 9(12).
               10  TXN-ACCT-TO               PIC 9(12).
               10  TXN-ACCT-TYPE-FROM        PIC X(2).
                   88  ACCT-CHECKING         VALUE 'CH'.
                   88  ACCT-SAVINGS          VALUE 'SV'.
                   88  ACCT-MONEY-MARKET     VALUE 'MM'.
                   88  ACCT-CD               VALUE 'CD'.
               10  TXN-ACCT-TYPE-TO          PIC X(2).

           05  TXN-AMOUNT-INFO.
               10  TXN-AMOUNT                PIC S9(13)V99 COMP-3.
               10  TXN-CURRENCY              PIC X(3).
               10  TXN-EXCHANGE-RATE         PIC 9(5)V9(6) COMP-3.
               10  TXN-LOCAL-AMOUNT          PIC S9(13)V99 COMP-3.
               10  TXN-FEE-AMOUNT            PIC S9(7)V99 COMP-3.

           05  TXN-DATETIME.
               10  TXN-DATE                  PIC 9(8).
               10  TXN-TIME                  PIC 9(6).
               10  TXN-POSTING-DATE          PIC 9(8).
               10  TXN-VALUE-DATE            PIC 9(8).
               10  TXN-TIMEZONE              PIC X(3).

           05  TXN-LOCATION.
               10  TXN-BRANCH-CODE           PIC 9(5).
               10  TXN-TERMINAL-ID           PIC X(8).
               10  TXN-MERCHANT-ID           PIC X(15).
               10  TXN-MCC-CODE              PIC 9(4).
               10  TXN-CITY                  PIC X(20).
               10  TXN-STATE                 PIC X(2).
               10  TXN-COUNTRY               PIC X(3).

           05  TXN-AUTH.
               10  TXN-AUTH-CODE             PIC X(6).
               10  TXN-AUTH-RESPONSE         PIC X(2).
                   88  AUTH-APPROVED         VALUE '00'.
                   88  AUTH-DECLINED         VALUE '05'.
                   88  AUTH-REFER            VALUE '01'.
               10  TXN-AUTH-TIME             PIC 9(6).
               10  TXN-CVV-RESULT            PIC X(1).
               10  TXN-AVS-RESULT            PIC X(1).

           05  TXN-DESCRIPTION.
               10  TXN-DESC-LINE1            PIC X(40).
               10  TXN-DESC-LINE2            PIC X(40).

           05  TXN-AUDIT.
               10  TXN-CREATED-BY            PIC X(8).
               10  TXN-CREATED-TIMESTAMP     PIC 9(14).
               10  TXN-MODIFIED-BY           PIC X(8).
               10  TXN-MODIFIED-TIMESTAMP    PIC 9(14).

           05  FILLER                        PIC X(11).
