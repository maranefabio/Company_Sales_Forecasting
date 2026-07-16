SELECT
	CONCAT(FtoFaturamento.Org, FtoFaturamento.Material) AS SKU,
	DimCanal.TipoCanal AS Channel,
	FtoFaturamento.DataFaturamento AS Dt,
	FtoFaturamento.QtdeUMR AS QT,
	CASE
	    WHEN FtoFaturamento.QtdeUMR > 0 THEN FtoFaturamento.ValLiq / FtoFaturamento.QtdeUMR
	    ELSE NULL
	END AS ASP
FROM
   	DW_BI_FK.dbo.FtoFaturamento AS FtoFaturamento
LEFT JOIN
    DW_BI_FK.dbo.DimCanal AS DimCanal ON
    FtoFaturamento.Cliente = DimCanal.Cliente
WHERE
    FtoFaturamento.TpFat <> 'ZL2B'