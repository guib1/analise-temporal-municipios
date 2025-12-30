from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


LOGGER = logging.getLogger(__name__)


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
	sys.path.insert(0, str(HERE))


@dataclass(frozen=True)
class ShapefileTarget:
	shapefile_path: Path
	codibge: Optional[str]

SUPPORTED_DISEASES = ("asma",)
INMET_OUTPUT_COLUMNS = [
	"codigo_ibge",
	"date",
	"globalradiation_max",
	"globalradiation_min",
	"globalradiation_mea",
	"precipitation_sum",
	"temperature_max",
	"temperature_min",
	"temperature_mea",
	"humidity_min",
	"humidity_mea",
	"wind_mea",
	"heatindex_max",
	"heatindex_min",
	"heatindex_mea",
]


def _parse_any_date(value: str) -> date:
	value = value.strip()
	# Accept YYYY-MM-DD or DD/MM/YYYY
	for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
		try:
			return datetime.strptime(value, fmt).date()
		except ValueError:
			continue
	# pandas fallback (handles other ISO forms)
	dt = pd.to_datetime(value, errors="raise")
	return dt.date()


def _format_ddmmyyyy(value: str | date) -> str:
	dt = _parse_any_date(value) if isinstance(value, str) else value
	return dt.strftime("%d/%m/%Y")


def _read_ibge_from_sidecar_csv(shp: Path) -> Optional[str]:
	directory = shp.parent
	for csv_file in sorted(directory.glob("*_ibge.csv")):
		try:
			df = pd.read_csv(csv_file)
		except Exception:
			continue
		for col in ("codigo_ibge", "cod_ibge", "COD_IBGE", "CD_MUN", "CD_GEOCMU"):
			if col in df.columns and not df.empty:
				try:
					return str(int(df[col].iloc[0])).zfill(7)
				except Exception:
					continue
	return None


def _read_ibge_from_shapefile(shp: Path) -> Optional[str]:
	try:
		import geopandas as gpd  # type: ignore
	except Exception:
		return None
	try:
		gdf = gpd.read_file(shp)
	except Exception:
		return None
	for col in ("code_muni", "CD_MUN", "CD_GEOCMU", "codigo_ibge", "COD_IBGE"):
		if col in gdf.columns and len(gdf) > 0:
			try:
				return str(int(gdf[col].iloc[0])).zfill(7)
			except Exception:
				continue
	return None


def infer_codibge(shapefile_path: Path) -> Optional[str]:
	code = _read_ibge_from_sidecar_csv(shapefile_path)
	if code:
		return code
	return _read_ibge_from_shapefile(shapefile_path)


def discover_shapefiles(shapefiles_dir: Path) -> List[ShapefileTarget]:
	shapefiles = sorted(shapefiles_dir.rglob("*.shp"))
	targets: List[ShapefileTarget] = []
	for shp in shapefiles:
		targets.append(ShapefileTarget(shapefile_path=shp, codibge=infer_codibge(shp)))
	return targets


def build_shapefile_catalog(shapefiles_dir: str | Path = "data/shapefiles") -> pd.DataFrame:
	"""
	Cria um catálogo (tabela) com os shapefiles disponíveis.

	Retorna um DataFrame com colunas:
	  - codibge (string, 7 dígitos quando possível)
	  - shapefile_path (string)
	  - shapefile_nome (string)
	"""
	root = Path(shapefiles_dir)
	targets = discover_shapefiles(root)
	rows: List[Dict[str, str]] = []
	for target in targets:
		shp = target.shapefile_path.resolve()
		codibge = target.codibge
		rows.append(
			{
				"codibge": str(codibge).zfill(7) if codibge and str(codibge).isdigit() else (codibge or ""),
				"shapefile_path": str(shp),
				"shapefile_nome": shp.stem,
			}
		)
	df = pd.DataFrame(rows)
	if df.empty:
		return df
	# Prefer known codibge first, then name for stability.
	df["__has_codibge"] = df["codibge"].astype(str).str.len().gt(0)
	df = df.sort_values(["__has_codibge", "codibge", "shapefile_nome"], ascending=[False, True, True]).drop(
		columns=["__has_codibge"]
	)
	return df.reset_index(drop=True)


def _load_module_from_path(module_name: str, path: Path):
	spec = importlib.util.spec_from_file_location(module_name, str(path))
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load module spec for {path}")
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module


def _standardize_frame(df: pd.DataFrame, codigo_ibge: str, *, source: str) -> pd.DataFrame:
	df = df.copy()

	# Standardize date column
	if "date" not in df.columns and "Data" in df.columns:
		df["date"] = df["Data"]
	if "date" not in df.columns:
		raise KeyError(f"{source}: output dataframe does not have a 'date' column")
	df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
	df = df.dropna(subset=["date"]).reset_index(drop=True)

	# Standardize IBGE column
	if "codigo_ibge" not in df.columns:
		for candidate in ("codigo_ibge", "COD_IBGE"):
			if candidate in df.columns:
				df["codigo_ibge"] = df[candidate]
				break
	df["codigo_ibge"] = str(codigo_ibge).zfill(7)

	# Deduplicate if needed
	key = ["codigo_ibge", "date"]
	if df.duplicated(subset=key).any():
		numeric_cols = [c for c in df.columns if c not in key and pd.api.types.is_numeric_dtype(df[c])]
		agg: Dict[str, str] = {}
		for c in numeric_cols:
			if c.endswith("_sum") or c in {"cases_sum", "deaths_sum"}:
				agg[c] = "sum"
			else:
				agg[c] = "mean"
		keep_cols = key + numeric_cols
		df = df[keep_cols].groupby(key, as_index=False).agg(agg)
	else:
		# Keep key first for merging
		pass

	return df


def _merge_sources(
	codigo_ibge: str,
	start: date,
	end: date,
	frames_by_source: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
	base = pd.DataFrame({"date": pd.date_range(start, end, freq="D").normalize()})
	base["codigo_ibge"] = str(codigo_ibge).zfill(7)
	combined = base

	for source, df in frames_by_source.items():
		if df is None or df.empty:
			continue
		df_std = _standardize_frame(df, codigo_ibge, source=source)
		# Avoid column name collisions: prefix only generic columns
		reserved = {"codigo_ibge", "date"}
		rename: Dict[str, str] = {}
		for c in df_std.columns:
			if c in reserved:
				continue
			if c in combined.columns:
				rename[c] = f"{source}_{c}"
		if rename:
			df_std = df_std.rename(columns=rename)
		combined = combined.merge(df_std, on=["codigo_ibge", "date"], how="left")

	# Final dedupe safety
	key = ["codigo_ibge", "date"]
	if combined.duplicated(subset=key).any():
		numeric_cols = [c for c in combined.columns if c not in key and pd.api.types.is_numeric_dtype(combined[c])]
		agg = {c: "mean" for c in numeric_cols}
		combined = combined.groupby(key, as_index=False).agg(agg)

	return combined.sort_values(["codigo_ibge", "date"]).reset_index(drop=True)


def load_schema_columns(schema_csv: str | Path) -> List[str]:
	"""Load expected output schema (column order) from a reference CSV header."""
	path = Path(schema_csv)
	if not path.exists():
		raise FileNotFoundError(f"Schema CSV not found: {path}")
	cols = pd.read_csv(path, nrows=0).columns.tolist()
	if not cols:
		raise ValueError(f"Schema CSV has no columns: {path}")
	return cols


def apply_output_schema(df: pd.DataFrame, schema_cols: Sequence[str]) -> pd.DataFrame:
	"""Force output to match `schema_cols` exactly (same columns + order; missing -> NA)."""
	out = df.copy()
	schema = list(schema_cols)
	if "date" not in out.columns:
		raise KeyError("Output dataframe is missing required 'date' column")

	# Key alignment: schema may use either 'codibge' or 'codigo_ibge'.
	if "codibge" in schema and "codibge" not in out.columns and "codigo_ibge" in out.columns:
		out = out.rename(columns={"codigo_ibge": "codibge"})
	if "codigo_ibge" in schema and "codigo_ibge" not in out.columns and "codibge" in out.columns:
		out = out.rename(columns={"codibge": "codigo_ibge"})

	# Reindex creates any missing columns (filled with NA) and drops extras in one go.
	return out.reindex(columns=schema, fill_value=pd.NA)


def _empty_time_index(start: date, end: date) -> pd.Series:
	return pd.Series(pd.date_range(start, end, freq="D").normalize(), name="date")


def _empty_schema_frame(
	*,
	start: date,
	end: date,
	codigo_ibge: str,
	columns: Sequence[str],
) -> pd.DataFrame:
	dates = _empty_time_index(start, end)
	out = pd.DataFrame({"codigo_ibge": str(codigo_ibge).zfill(7), "date": dates})
	for col in columns:
		if col in {"codigo_ibge", "date"}:
			continue
		out[col] = pd.NA
	return out


def _write_source_csv(
	*,
	df: pd.DataFrame,
	out_csv: Path,
	codigo_ibge: str,
	source: str,
	start: date,
	end: date,
	expected_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
	"""
	Garante que o CSV por fonte sempre exista.
	- Se df vazio/None: escreve placeholder (datas + codigo_ibge) e colunas esperadas (se fornecidas).
	- Se df não vazio: padroniza para 'codigo_ibge' + 'date' e salva.
	"""
	if df is None or df.empty:
		cols = expected_columns if expected_columns is not None else ("codigo_ibge", "date")
		df_out = _empty_schema_frame(start=start, end=end, codigo_ibge=codigo_ibge, columns=cols)
	else:
		df_out = _standardize_frame(df, codigo_ibge, source=source)
		# Ensure key order
		ordered = ["codigo_ibge", "date"] + [c for c in df_out.columns if c not in {"codigo_ibge", "date"}]
		df_out = df_out[ordered]

	out_csv.parent.mkdir(parents=True, exist_ok=True)
	df_out.to_csv(out_csv, index=False)
	return df_out


def run_all_for_target(
	target: ShapefileTarget,
	*,
	start: date,
	end: date,
	output_dir: Path,
	cache_dir: Path,
	disease: str = "asma",
) -> Tuple[str, Dict[str, Path], pd.DataFrame]:
	shp = target.shapefile_path
	codibge = target.codibge or shp.stem
	codibge_norm = str(codibge).zfill(7) if str(codibge).isdigit() else str(codibge)
	disease_norm = str(disease).strip().lower()
	if disease_norm not in SUPPORTED_DISEASES:
		raise ValueError(f"Doença não suportada: {disease!r}. Suportadas: {', '.join(SUPPORTED_DISEASES)}")

	outputs: Dict[str, Path] = {}
	frames: Dict[str, pd.DataFrame] = {}

	# Ensure directories
	(output_dir / "cetesb").mkdir(parents=True, exist_ok=True)
	(output_dir / "inmet").mkdir(parents=True, exist_ok=True)
	(output_dir / "era5").mkdir(parents=True, exist_ok=True)
	(output_dir / "merra2").mkdir(parents=True, exist_ok=True)
	(output_dir / "tropomi").mkdir(parents=True, exist_ok=True)
	(output_dir / "modis").mkdir(parents=True, exist_ok=True)
	(output_dir / "diversos").mkdir(parents=True, exist_ok=True)
	(output_dir / "datasus").mkdir(parents=True, exist_ok=True)

	# CETESB (requires SP)
	try:
		mod = _load_module_from_path("CETESB_module", HERE / "CETESB.py")

		out_csv = output_dir / "cetesb" / f"cetesb_{codibge_norm}.csv"
		df_raw = mod.CETESBDownloader().fetch_data(
			shapefile_path=str(shp),
			start_date=_format_ddmmyyyy(start),
			end_date=_format_ddmmyyyy(end),
			output_csv=str(out_csv),
		)
		outputs["cetesb"] = out_csv
		frames["cetesb"] = _write_source_csv(
			df=df_raw,
			out_csv=out_csv,
			codigo_ibge=codibge_norm,
			source="cetesb",
			start=start,
			end=end,
		)
	except Exception as exc:
		LOGGER.warning("CETESB falhou para %s: %s", shp, exc)
		out_csv = output_dir / "cetesb" / f"cetesb_{codibge_norm}.csv"
		outputs["cetesb"] = out_csv
		frames["cetesb"] = _write_source_csv(
			df=pd.DataFrame(),
			out_csv=out_csv,
			codigo_ibge=codibge_norm,
			source="cetesb",
			start=start,
			end=end,
		)

	# INMET
	try:
		mod = _load_module_from_path("INMET_module", HERE / "INMET.py")

		out_csv = output_dir / "inmet" / f"inmet_{codibge_norm}.csv"
		df_raw = mod.INMETDownloader().fetch_daily_data(
			shapefile_path=str(shp),
			start=str(start),
			end=str(end),
			out_csv=str(out_csv),
		)
		outputs["inmet"] = out_csv
		frames["inmet"] = _write_source_csv(
			df=df_raw,
			out_csv=out_csv,
			codigo_ibge=codibge_norm,
			source="inmet",
			start=start,
			end=end,
			expected_columns=INMET_OUTPUT_COLUMNS,
		)
	except Exception as exc:
		LOGGER.warning("INMET falhou para %s: %s", shp, exc)
		out_csv = output_dir / "inmet" / f"inmet_{codibge_norm}.csv"
		outputs["inmet"] = out_csv
		frames["inmet"] = _write_source_csv(
			df=pd.DataFrame(),
			out_csv=out_csv,
			codigo_ibge=codibge_norm,
			source="inmet",
			start=start,
			end=end,
			expected_columns=INMET_OUTPUT_COLUMNS,
		)

	# ERA5
	try:
		mod = _load_module_from_path("ERA5_module", HERE / "ERA5.py")

		out_csv = output_dir / "era5" / f"era5_{codibge_norm}.csv"
		out_nc = output_dir / "era5" / f"temp_era5_{codibge_norm}.nc"
		df_raw = mod.ERA5Downloader().fetch_daily_data(
			shapefile_path=str(shp),
			start=str(start),
			end=str(end),
			out_nc=str(out_nc),
			out_csv=str(out_csv),
		)
		outputs["era5"] = out_csv
		frames["era5"] = _write_source_csv(
			df=df_raw,
			out_csv=out_csv,
			codigo_ibge=codibge_norm,
			source="era5",
			start=start,
			end=end,
		)
	except Exception as exc:
		LOGGER.warning("ERA5 falhou para %s: %s", shp, exc)
		out_csv = output_dir / "era5" / f"era5_{codibge_norm}.csv"
		outputs["era5"] = out_csv
		frames["era5"] = _write_source_csv(
			df=pd.DataFrame(),
			out_csv=out_csv,
			codigo_ibge=codibge_norm,
			source="era5",
			start=start,
			end=end,
		)

	# MERRA2
	try:
		mod = _load_module_from_path("MERRA2_module", HERE / "MERRA2.py")

		out_csv = output_dir / "merra2" / f"merra2_{codibge_norm}.csv"
		df_raw = mod.MERRA2Downloader().fetch_daily_data(
			shapefile_path=str(shp),
			start=str(start),
			end=str(end),
			out_csv=str(out_csv),
		)
		outputs["merra2"] = out_csv
		frames["merra2"] = _write_source_csv(
			df=df_raw,
			out_csv=out_csv,
			codigo_ibge=codibge_norm,
			source="merra2",
			start=start,
			end=end,
		)
	except Exception as exc:
		LOGGER.warning("MERRA2 falhou para %s: %s", shp, exc)
		out_csv = output_dir / "merra2" / f"merra2_{codibge_norm}.csv"
		outputs["merra2"] = out_csv
		frames["merra2"] = _write_source_csv(
			df=pd.DataFrame(),
			out_csv=out_csv,
			codigo_ibge=codibge_norm,
			source="merra2",
			start=start,
			end=end,
		)

	# TROPOMI (expects DD/MM/YYYY)
	try:
		mod = _load_module_from_path("TROPOMI_module", HERE / "TROPOMI.py")

		out_csv = output_dir / "tropomi" / f"tropomi_{codibge_norm}.csv"
		cache_subdir = cache_dir / "tropomi" / codibge_norm
		cache_subdir.mkdir(parents=True, exist_ok=True)
		df_raw = mod.TropomiDownloader().fetch_data(
			shapefile_path=str(shp),
			start_date=_format_ddmmyyyy(start),
			end_date=_format_ddmmyyyy(end),
			output_csv=str(out_csv),
			cache_dir=str(cache_subdir),
		)
		outputs["tropomi"] = out_csv
		frames["tropomi"] = _write_source_csv(
			df=df_raw,
			out_csv=out_csv,
			codigo_ibge=codibge_norm,
			source="tropomi",
			start=start,
			end=end,
		)
	except Exception as exc:
		LOGGER.warning("TROPOMI falhou para %s: %s", shp, exc)
		out_csv = output_dir / "tropomi" / f"tropomi_{codibge_norm}.csv"
		outputs["tropomi"] = out_csv
		frames["tropomi"] = _write_source_csv(
			df=pd.DataFrame(),
			out_csv=out_csv,
			codigo_ibge=codibge_norm,
			source="tropomi",
			start=start,
			end=end,
		)

	# MODIS (expects DD/MM/YYYY)
	try:
		mod = _load_module_from_path("MODIS_module", HERE / "MODIS.py")

		out_csv = output_dir / "modis" / f"modis_{codibge_norm}.csv"
		df_raw = mod.ModisDownloader().fetch_data(
			shapefile_path=str(shp),
			start_date=_format_ddmmyyyy(start),
			end_date=_format_ddmmyyyy(end),
			output_csv=str(out_csv),
		)
		outputs["modis"] = out_csv
		frames["modis"] = _write_source_csv(
			df=df_raw,
			out_csv=out_csv,
			codigo_ibge=codibge_norm,
			source="modis",
			start=start,
			end=end,
		)
	except Exception as exc:
		LOGGER.warning("MODIS falhou para %s: %s", shp, exc)
		out_csv = output_dir / "modis" / f"modis_{codibge_norm}.csv"
		outputs["modis"] = out_csv
		frames["modis"] = _write_source_csv(
			df=pd.DataFrame(),
			out_csv=out_csv,
			codigo_ibge=codibge_norm,
			source="modis",
			start=start,
			end=end,
		)

	if str(codibge_norm).isdigit():
		try:
			mod = _load_module_from_path("DATASUS_module", HERE / "DATASUS.py")

			out_csv = output_dir / "datasus" / f"datasus_{disease_norm}_daily_{codibge_norm}.csv"
			if disease_norm == "asma":
				df_raw = mod.DataSUSDownloader().fetch_sih_asthma_daily(
					cod_ibge=codibge_norm,
					start=start,
					end=end,
					output_csv=out_csv,
				)
			else:  # pragma: no cover - guarded by SUPPORTED_DISEASES
				raise ValueError(f"Doença não suportada: {disease_norm!r}")
			outputs["datasus"] = out_csv
			frames["datasus"] = _write_source_csv(
				df=df_raw,
				out_csv=Path(out_csv),
				codigo_ibge=codibge_norm,
				source="datasus",
				start=start,
				end=end,
			)
		except Exception as exc:
			LOGGER.warning("DATASUS falhou para %s (%s): %s", shp, codibge_norm, exc)
			out_csv = output_dir / "datasus" / f"datasus_{disease_norm}_daily_{codibge_norm}.csv"
			outputs["datasus"] = out_csv
			frames["datasus"] = _write_source_csv(
				df=pd.DataFrame(),
				out_csv=out_csv,
				codigo_ibge=codibge_norm,
				source="datasus",
				start=start,
				end=end,
			)

	# Diversos / INDICE-CALCULADO (precisa rodar por último)
	try:
		diversos_path = HERE / "INDICE-CALCULADO.py"
		mod = _load_module_from_path("indice_calculado", diversos_path)
		out_csv = output_dir / "diversos" / f"diversos_{codibge_norm}.csv"
		inmet_df = frames.get("inmet", pd.DataFrame())
		df_raw = mod.DiversosDownloader().fetch_data(
			shapefile_path=str(shp),
			start_date=_format_ddmmyyyy(start),
			end_date=_format_ddmmyyyy(end),
			output_csv=str(out_csv),
			inmet_df=inmet_df,
		)
		outputs["diversos"] = out_csv
		frames["diversos"] = _write_source_csv(
			df=df_raw,
			out_csv=out_csv,
			codigo_ibge=codibge_norm,
			source="diversos",
			start=start,
			end=end,
		)
	except Exception as exc:
		LOGGER.warning("DIVERSOS falhou para %s: %s", shp, exc)
		out_csv = output_dir / "diversos" / f"diversos_{codibge_norm}.csv"
		outputs["diversos"] = out_csv
		frames["diversos"] = _write_source_csv(
			df=pd.DataFrame(),
			out_csv=out_csv,
			codigo_ibge=codibge_norm,
			source="diversos",
			start=start,
			end=end,
		)

	combined = _merge_sources(codibge_norm, start, end, frames)
	return codibge_norm, outputs, combined


def download_all(
	*,
	start: str | date,
	end: str | date,
	disease: str = "asma",
	shapefiles_dir: str | Path = "data/shapefiles",
	shapefile: str | Path | None = None,
	shapefiles: Sequence[str | Path] | None = None,
	output_dir: str | Path = "data/output",
	cache_dir: str | Path = "data/cache",
	final_csv: str | Path = "data/output/final/final_by_ibge_date.csv",
	write_per_municipio: bool = True,
	write_final: bool = True,
	final_schema: str = "reference",
	schema_csv: str | Path | None = None,
	log_level: str = "INFO",
) -> pd.DataFrame:
	"""
	Orquestra o download/processamento de todas as fontes para um ou mais municípios.

	Parâmetros principais
	--------------------
	start, end:
		Datas (YYYY-MM-DD, DD/MM/YYYY, ou datetime.date).
	disease:
		Doença para o módulo DataSUS (por enquanto apenas 'asma').
	shapefile:
		Se informado, processa apenas este shapefile.
	shapefiles:
		Lista de shapefiles. Se informado, tem prioridade sobre shapefiles_dir.
	shapefiles_dir:
		Diretório raiz para buscar *.shp recursivamente.
	output_dir:
		Diretório base onde serão salvos os CSVs por fonte e os finais.
	cache_dir:
		Diretório de cache usado (por ex. TROPOMI por município).
	final_csv:
		Caminho do CSV final unificado (codibge+date).
	final_schema:
		"all" para todas as colunas unificadas,
		"inmet" para forçar o schema/ordem do INMET,
		"reference" para forçar o schema/ordem a partir de `schema_csv`.
	schema_csv:
		Caminho de um CSV de referência (somente o header é usado) para definir colunas e ordem.

	Retorna
	-------
	pandas.DataFrame
		DataFrame final unificado (codibge+date).
	"""
	# Configure logging only if the host app hasn't configured it yet.
	root_logger = logging.getLogger()
	if not root_logger.handlers:
		logging.basicConfig(
			level=getattr(logging, log_level.upper(), logging.INFO),
			format="%(asctime)s %(levelname)s %(name)s - %(message)s",
		)

	start_date = _parse_any_date(start) if isinstance(start, str) else start
	end_date = _parse_any_date(end) if isinstance(end, str) else end
	if end_date < start_date:
		raise ValueError("end deve ser >= start")

	output_dir_path = Path(output_dir)
	cache_dir_path = Path(cache_dir)
	final_csv_path = Path(final_csv)
	output_dir_path.mkdir(parents=True, exist_ok=True)
	cache_dir_path.mkdir(parents=True, exist_ok=True)
	final_csv_path.parent.mkdir(parents=True, exist_ok=True)

	final_schema_norm = str(final_schema).strip().lower()
	if final_schema_norm not in {"all", "inmet", "reference"}:
		raise ValueError("final_schema must be 'all', 'inmet', or 'reference'")

	schema_cols: Optional[List[str]] = None
	if final_schema_norm == "reference":
		if schema_csv is None:
			raise ValueError("final_schema='reference' requires schema_csv=<path-to-reference-csv>")
		schema_cols = load_schema_columns(schema_csv)

	# Resolve targets
	targets: List[ShapefileTarget] = []
	if shapefiles is not None:
		for shp in shapefiles:
			shp_path = Path(shp)
			targets.append(ShapefileTarget(shapefile_path=shp_path, codibge=infer_codibge(shp_path)))
	elif shapefile is not None:
		shp_path = Path(shapefile)
		targets.append(ShapefileTarget(shapefile_path=shp_path, codibge=infer_codibge(shp_path)))
	else:
		targets = discover_shapefiles(Path(shapefiles_dir))

	if not targets:
		LOGGER.warning("Nenhum shapefile encontrado para processar.")
		return pd.DataFrame()

	all_frames: List[pd.DataFrame] = []
	for target in targets:
		LOGGER.info("Processando %s", target.shapefile_path)
		codibge, _outputs, combined = run_all_for_target(
			target,
			start=start_date,
			end=end_date,
			output_dir=output_dir_path,
			cache_dir=cache_dir_path,
			disease=disease,
		)
		combined_internal = combined
		combined_out = combined_internal
		if final_schema_norm == "inmet":
			for col in INMET_OUTPUT_COLUMNS:
				if col not in combined_out.columns:
					combined_out[col] = pd.NA
			combined_out = combined_out[INMET_OUTPUT_COLUMNS]
		elif final_schema_norm == "reference" and schema_cols is not None:
			# Only apply schema to the per-municipio CSV. We'll apply schema to df_all at the end.
			combined_out = apply_output_schema(combined_out, schema_cols)

		if write_per_municipio:
			out_per_muni = output_dir_path / "final" / "by_municipio" / f"final_{codibge}.csv"
			out_per_muni.parent.mkdir(parents=True, exist_ok=True)
			combined_out.to_csv(out_per_muni, index=False)
			LOGGER.info("CSV final do município gerado -> %s", out_per_muni)
		# Keep internal frames in a stable key for the global merge/groupby.
		all_frames.append(combined_internal)

	df_all = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
	if df_all.empty:
		LOGGER.warning("Nenhum dado foi gerado.")
		return df_all

	# Groupby final (segurança contra duplicatas)
	key = ["codigo_ibge", "date"]
	df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce").dt.normalize()
	df_all = df_all.dropna(subset=["date"]).reset_index(drop=True)
	numeric_cols = [c for c in df_all.columns if c not in key and pd.api.types.is_numeric_dtype(df_all[c])]
	agg = {c: "mean" for c in numeric_cols}
	df_all = df_all.groupby(key, as_index=False).agg(agg).sort_values(key).reset_index(drop=True)

	if final_schema_norm == "inmet":
		# Ensure INMET columns exist and are in the expected order.
		for col in INMET_OUTPUT_COLUMNS:
			if col not in df_all.columns:
				df_all[col] = pd.NA
		df_all = df_all[INMET_OUTPUT_COLUMNS]
	elif final_schema_norm == "reference" and schema_cols is not None:
		df_all = apply_output_schema(df_all, schema_cols)

	if write_final:
		df_all.to_csv(final_csv_path, index=False)
		LOGGER.info("CSV final unificado gerado -> %s", final_csv_path)

	return df_all