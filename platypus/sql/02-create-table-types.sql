USE [BigData];
---
IF OBJECT_ID('Typen', 'U') IS NULL
    BEGIN
        CREATE TABLE [dbo].[Typen]
        (
            [TYP]          [nvarchar](10)  NOT NULL,
            [BESCHREIBUNG] [nvarchar](255) NOT NULL,
            CONSTRAINT [PK_Typen] PRIMARY KEY CLUSTERED
                ([TYP] ASC) ON [PRIMARY]
        ) ON [PRIMARY]

        INSERT INTO [dbo].[Typen] ([TYP], [BESCHREIBUNG])
        VALUES ('BRW', 'Bodenrichtwert'),
               ('IRW', 'Immobilienrichtwert')
    END;
