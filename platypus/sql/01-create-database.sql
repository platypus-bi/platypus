USE master;
---
IF DB_ID(N'BigData') IS NULL
    BEGIN
        CREATE DATABASE [BigData]
    END;
