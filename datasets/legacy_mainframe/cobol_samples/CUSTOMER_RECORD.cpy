      *****************************************************************
      * CUSTOMER MASTER FILE RECORD LAYOUT
      * This copybook defines the standard customer record structure
      * used in the core banking system since 1987.
      *
      * Record Type: Fixed-length 500 bytes
      * Character Set: EBCDIC
      * Last Modified: 2024-01-15
      *****************************************************************
       01  CUSTOMER-MASTER-RECORD.
           05  CUST-HEADER.
               10  CUST-RECORD-TYPE          PIC X(2).
                   88  CUST-ACTIVE           VALUE 'AC'.
                   88  CUST-INACTIVE         VALUE 'IN'.
                   88  CUST-CLOSED           VALUE 'CL'.
               10  CUST-RECORD-VERSION       PIC 9(2).
               10  CUST-TIMESTAMP            PIC 9(14).
               10  FILLER                    PIC X(2).

           05  CUST-IDENTITY.
               10  CUST-ID                   PIC 9(10).
               10  CUST-TYPE                 PIC X(1).
                   88  CUST-INDIVIDUAL       VALUE 'I'.
                   88  CUST-CORPORATE        VALUE 'C'.
                   88  CUST-GOVERNMENT       VALUE 'G'.
               10  CUST-SSN                  PIC 9(9).
               10  CUST-TAX-ID               PIC X(15).

           05  CUST-NAME-INFO.
               10  CUST-LAST-NAME            PIC X(30).
               10  CUST-FIRST-NAME           PIC X(20).
               10  CUST-MIDDLE-INIT          PIC X(1).
               10  CUST-TITLE                PIC X(5).
               10  CUST-SUFFIX               PIC X(5).

           05  CUST-ADDRESS.
               10  CUST-ADDR-LINE1           PIC X(40).
               10  CUST-ADDR-LINE2           PIC X(40).
               10  CUST-CITY                 PIC X(25).
               10  CUST-STATE                PIC X(2).
               10  CUST-ZIP-CODE             PIC X(10).
               10  CUST-COUNTRY              PIC X(3).

           05  CUST-CONTACT.
               10  CUST-PHONE-HOME           PIC 9(10).
               10  CUST-PHONE-WORK           PIC 9(10).
               10  CUST-PHONE-MOBILE         PIC 9(10).
               10  CUST-EMAIL                PIC X(50).

           05  CUST-FINANCIAL.
               10  CUST-CREDIT-SCORE         PIC 9(3).
               10  CUST-CREDIT-LIMIT         PIC S9(9)V99 COMP-3.
               10  CUST-BALANCE              PIC S9(11)V99 COMP-3.
               10  CUST-AVAILABLE-CREDIT     PIC S9(9)V99 COMP-3.
               10  CUST-LAST-PAYMENT-AMT     PIC S9(9)V99 COMP-3.
               10  CUST-LAST-PAYMENT-DATE    PIC 9(8).

           05  CUST-RELATIONSHIP.
               10  CUST-OPEN-DATE            PIC 9(8).
               10  CUST-BRANCH-CODE          PIC 9(5).
               10  CUST-OFFICER-ID           PIC 9(6).
               10  CUST-SEGMENT              PIC X(2).
                   88  CUST-PREMIUM          VALUE 'PR'.
                   88  CUST-STANDARD         VALUE 'ST'.
                   88  CUST-BASIC            VALUE 'BA'.
               10  CUST-RISK-RATING          PIC 9(1).

           05  CUST-FLAGS.
               10  CUST-KYC-VERIFIED         PIC X(1).
                   88  KYC-YES               VALUE 'Y'.
                   88  KYC-NO                VALUE 'N'.
               10  CUST-AML-FLAG             PIC X(1).
               10  CUST-FRAUD-FLAG           PIC X(1).
               10  CUST-DECEASED-FLAG        PIC X(1).
               10  CUST-BANKRUPT-FLAG        PIC X(1).
               10  FILLER                    PIC X(5).

           05  FILLER                        PIC X(43).
