USE [BigData];
---
IF OBJECT_ID('Datenbestand', 'U') IS NULL
    BEGIN
        CREATE TABLE [dbo].[Datenbestand]
        (
            [JAHR]         [int]          NOT NULL,
            [TYP]          [nvarchar](10) NOT NULL,
            [AKTUALISIERT] [datetime]     NOT NULL,
            CONSTRAINT [PK_Datenbestand] PRIMARY KEY CLUSTERED
                ([JAHR] ASC, [TYP] ASC) ON [PRIMARY]
        ) ON [PRIMARY]


        ALTER TABLE [dbo].[Datenbestand]
            WITH CHECK ADD CONSTRAINT [FK_Datenbestand_Typen] FOREIGN KEY ([TYP])
                REFERENCES [dbo].[Typen] ([TYP])


        ALTER TABLE [dbo].[Datenbestand]
            CHECK CONSTRAINT [FK_Datenbestand_Typen]
    END;

