from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.utils.geo import parse_date, format_ddmmyyyy, get_ibge_code_str


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
	"""Alias kept for backward compatibility within this module."""
	return parse_date(value)


def _format_ddmmyyyy(value: str | date) -> str:
	"""Alias kept for backward compatibility within this module."""
	return format_ddmmyyyy(value)


def infer_codibge(shapefile_path: Path) -> Optional[str]:
	"""Return the zero-padded 7-digit IBGE code for a shapefile."""
	return get_ibge_code_str(shapefile_path)


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



import concurrent.futures
import traceback

class DownloadOrchestrator:
	def __init__(self, max_workers: int = 3):
		self.max_workers = max_workers
		# Mapping: key -> (module_name, file_name, class_name, method_name, expects_out_csv_arg)
		# We'll handle arguments dynamically based on the key
		pass

	def _run_scraper(self, key: str, target: ShapefileTarget, start: date, end: date, output_dir: Path, cache_dir: Path, disease: str, context: dict = None) -> pd.DataFrame:
		"""
		Executa um scraper específico de forma isolada e segura.
		Retorna o DataFrame resultante (ou vazio em caso de erro).
		"""
		shp = target.shapefile_path
		codibge = target.codibge or shp.stem
		codibge_norm = str(codibge).zfill(7) if str(codibge).isdigit() else str(codibge)
		
		# Define paths
		out_csv = output_dir / key / f"{key}_{codibge_norm}.csv"
		cache_subdir = cache_dir / key / codibge_norm
		
		# Ensure dirs
		out_csv.parent.mkdir(parents=True, exist_ok=True)
		cache_subdir.mkdir(parents=True, exist_ok=True)

		LOGGER.info(f"[{key.upper()}] Iniciando para {codibge_norm}...")
		
		try:
			df = pd.DataFrame()
			
			# Dispatch logic based on key
			if key == "cetesb":
				mod = _load_module_from_path("CETESB_module", HERE / "CETESB.py")
				df = mod.CETESBDownloader().fetch_data(
					shapefile_path=str(shp),
					start_date=_format_ddmmyyyy(start),
					end_date=_format_ddmmyyyy(end),
					output_csv=str(out_csv)
				)
			
			elif key == "inmet":
				mod = _load_module_from_path("INMET_module", HERE / "INMET.py")
				df = mod.INMETDownloader().fetch_daily_data(
					shapefile_path=str(shp),
					start=str(start),
					end=str(end),
					out_csv=str(out_csv)
				)
				# INMET scraper returns None sometimes, but writes file. Reload if needed.
				if df is None and out_csv.exists():
				    df = pd.read_csv(out_csv)
			
			elif key == "era5":
				mod = _load_module_from_path("ERA5_module", HERE / "ERA5.py")
				out_nc = cache_subdir / f"temp_era5_{codibge_norm}.nc"
				df = mod.ERA5Downloader().fetch_daily_data(
					shapefile_path=str(shp),
					start=str(start),
					end=str(end),
					out_nc=str(out_nc),
					out_csv=str(out_csv)
				)
				# Cleanup NetCDF
				if out_nc.exists():
					try:
						out_nc.unlink()
					except Exception:
						pass

			elif key == "merra2":
				mod = _load_module_from_path("MERRA2_module", HERE / "MERRA2.py")
				df = mod.MERRA2Downloader().fetch_daily_data(
					shapefile_path=str(shp),
					start=str(start),
					end=str(end),
					out_csv=str(out_csv)
				)

			elif key == "tropomi":
				mod = _load_module_from_path("TROPOMI_module", HERE / "TROPOMI.py")
				df = mod.TropomiDownloader().fetch_data(
					shapefile_path=str(shp),
					start_date=_format_ddmmyyyy(start),
					end_date=_format_ddmmyyyy(end),
					output_csv=str(out_csv),
					cache_dir=str(cache_subdir)
				)
			
			elif key == "modis":
				mod = _load_module_from_path("MODIS_module", HERE / "MODIS.py")
				df = mod.ModisDownloader().fetch_data(
					shapefile_path=str(shp),
					start_date=_format_ddmmyyyy(start),
					end_date=_format_ddmmyyyy(end),
					output_csv=str(out_csv),
					cache_dir=str(cache_subdir)
				)

			elif key == "omi":
				mod = _load_module_from_path("OMI_module", HERE / "OMI.py")
				df = mod.OMIDownloader().fetch_data(
					shapefile_path=str(shp),
					start_date=_format_ddmmyyyy(start),
					end_date=_format_ddmmyyyy(end),
					output_csv=str(out_csv),
					cache_dir=str(cache_subdir)
				)

			elif key == "datasus":
				if not str(codibge_norm).isdigit():
					LOGGER.warning(f"[{key.upper()}] Pular: código IBGE inválido {codibge_norm}")
					return pd.DataFrame()
					
				mod = _load_module_from_path("DATASUS_module", HERE / "DATASUS.py")
				disease_norm = disease.strip().lower()
				# Special filename for DATASUS
				out_csv = output_dir / "datasus" / f"datasus_{disease_norm}_daily_{codibge_norm}.csv"
				
				if disease_norm == "asma":
					df = mod.DataSUSDownloader().fetch_sih_asthma_daily(
						cod_ibge=codibge_norm,
						start=start,
						end=end,
						output_csv=out_csv
					)
				else:
					raise ValueError(f"Doença não suportada: {disease}")

			elif key == "indices":
				# Dependente do INMET
				inmet_df = context.get("inmet", pd.DataFrame())
				if inmet_df.empty and (output_dir / "inmet" / f"inmet_{codibge_norm}.csv").exists():
					try:
						inmet_df = pd.read_csv(output_dir / "inmet" / f"inmet_{codibge_norm}.csv")
					except:
						pass
				
				# Load module (filename is still INDICE-CALCULADO.py)
				mod = _load_module_from_path("INDICE_CALCULADO_module", HERE / "INDICE-CALCULADO.py")
				
				# Output file: indices_{codibge}.csv
				out_csv = output_dir / "indices" / f"indices_{codibge_norm}.csv"
				
				df = mod.DiversosDownloader().fetch_data(
					shapefile_path=str(shp),
					start_date=_format_ddmmyyyy(start),
					end_date=_format_ddmmyyyy(end),
					output_csv=str(out_csv),
					inmet_df=inmet_df
				)

			else:
				LOGGER.error(f"Scraper desconhecido: {key}")
				return pd.DataFrame()

			# Standardization and saving
			expected_cols = INMET_OUTPUT_COLUMNS if key == "inmet" else None
			return _write_source_csv(
				df=df,
				out_csv=out_csv,
				codigo_ibge=codibge_norm,
				source=key,
				start=start,
				end=end,
				expected_columns=expected_cols
			)

		except Exception as exc:
			LOGGER.error(f"[{key.upper()}] Erro ao processar {codibge_norm}: {exc}")
			LOGGER.debug(traceback.format_exc())
			
			# Ensure empty file is written to avoid pipeline breakage
			return _write_source_csv(
				df=pd.DataFrame(),
				out_csv=out_csv,
				codigo_ibge=codibge_norm,
				source=key,
				start=start,
				end=end,
				expected_columns=INMET_OUTPUT_COLUMNS if key == "inmet" else None
			)

	def process_municipio(
		self,
		target: ShapefileTarget,
		start: date,
		end: date,
		output_dir: Path,
		cache_dir: Path,
		disease: str
	) -> Tuple[str, Dict[str, Path], pd.DataFrame]:
		"""Orquestra o download paralelo para um único município."""
		
		LOGGER.info("=== Processando Município: %s ===", target.shapefile_path.stem)
		codibge = target.codibge
		shapefile_path = target.shapefile_path
		
		# Tentar carregar código IBGE autoritativo do arquivo CSV auxiliar
		# Ex: SP_São_Paulo_ibge.csv na mesma pasta do shapefile
		auth_codibge = codibge
		ibge_csv = shapefile_path.parent / f"{shapefile_path.stem}_ibge.csv"
		if ibge_csv.exists():
			try:
				idf = pd.read_csv(ibge_csv, dtype=str)
				if not idf.empty:
					 # Pega a primeira coluna (assumindo que seja o codigo)
					auth_codibge = idf.iloc[0, 0]
					LOGGER.info(f"Código IBGE autoritativo carregado de {ibge_csv.name}: {auth_codibge}")
			except Exception as e:
				LOGGER.warning(f"Erro ao ler {ibge_csv.name}: {e}")
		
		codibge_norm = str(auth_codibge).zfill(7) if str(auth_codibge).isdigit() else str(auth_codibge)

		outputs = {}
		frames = {}
		
		# Tier 1: Independent Tasks
		# INMET is in Tier 1 but required for Tier 2
		tier1_keys = ["cetesb", "inmet", "era5", "merra2", "tropomi", "modis", "omi", "datasus"]
		
		# TQDM for visualization (auto-detects notebook/console)
		try:
			from tqdm.auto import tqdm
			USE_TQDM = True
		except ImportError:
			USE_TQDM = False

		progress_bars = {}
		if USE_TQDM:
			print(f"\nIniciando downloads paralelos para {codibge_norm} ({len(tier1_keys)} fontes)...")
			# Create bars in fixed order so they don't jump around
			for i, key in enumerate(tier1_keys):
				# Indeterminate bar since we don't know exact chunks yet
				progress_bars[key] = tqdm(total=1, desc=f"[{key.upper()}] Aguardando", position=i, leave=True)

		with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
			future_to_key = {}
			for key in tier1_keys:
				if USE_TQDM:
					progress_bars[key].set_description(f"[{key.upper()}] Processando...")
				
				future = executor.submit(self._run_scraper, key, target, start, end, output_dir, cache_dir, disease)
				future_to_key[future] = key
			
			for future in concurrent.futures.as_completed(future_to_key):
				key = future_to_key[future]
				try:
					df_res = future.result()
					frames[key] = df_res
					
					# Determine output path logic (simplified)
					shp = target.shapefile_path
					codibge = target.codibge or shp.stem
					codibge_norm = str(codibge).zfill(7) if str(codibge).isdigit() else str(codibge)
					
					if key == "datasus":
						disease_norm = disease.strip().lower()
						outputs[key] = output_dir / "datasus" / f"datasus_{disease_norm}_daily_{codibge_norm}.csv"
					else:
						outputs[key] = output_dir / key / f"{key}_{codibge_norm}.csv"
						
					msg = f"[{key.upper()}] Finalizado."
					LOGGER.info(msg)
					
					if USE_TQDM:
						progress_bars[key].set_description(f"[{key.upper()}] Concluído")
						progress_bars[key].update(1)
						progress_bars[key].close()

				except Exception as exc:
					err_msg = f"[{key.upper()}] Falha: {exc}"
					LOGGER.error(err_msg)
					if USE_TQDM:
						progress_bars[key].set_description(f"[{key.upper()}] Erro")
						progress_bars[key].close()
		
		# Tier 2: Dependent Tasks (Indices Calculados need INMET)
		# We run this sequentially after Tier 1 to ensure INMET is ready
		LOGGER.info("[INDICES] Iniciando cálculo de índices (depende do INMET)...")
		frames["indices"] = self._run_scraper(
			"indices", target, start, end, output_dir, cache_dir, disease, context={"inmet": frames.get("inmet")}
		)
		
		# Merge all
		shp = target.shapefile_path
		
		# Use authoritative codibge if found, else target
		final_cod = auth_codibge if auth_codibge else (target.codibge or shp.stem)
		codibge_norm = str(final_cod).zfill(7) if str(final_cod).isdigit() else str(final_cod)
		
		# Enrich frames with authoritative IBGE code BEFORE merge/dedupe
		for k, df_frame in frames.items():
			if df_frame is not None and not df_frame.empty:
				df_frame["codigo_ibge"] = codibge_norm
		
		combined = _merge_sources(codibge_norm, start, end, frames)
		
		# Limpeza de Cache (Cleanup)
		# Se tudo correu bem e cache_subdir foi definido, apagamos o cache deste município
		try:
			if 'cache_subdir' in locals() and cache_subdir.exists():
				import shutil
				shutil.rmtree(cache_subdir)
				LOGGER.info(f"Cache limpo com sucesso: {cache_subdir}")
		except Exception as e:
			if 'cache_subdir' in locals():
				LOGGER.warning(f"Não foi possível limpar o cache {cache_subdir}: {e}")

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
	max_workers: int = 3
) -> pd.DataFrame:
	"""
	Orquestra o download/processamento de todas as fontes para um ou mais municípios.
	"""
	# Configure logging 
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

	# Schema validation
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
		root = Path(shapefiles_dir)
		if root.exists():
			targets = discover_shapefiles(root)
		else:
			LOGGER.warning(f"Diretório de shapefiles não encontrado: {root}")

	if not targets:
		LOGGER.warning("Nenhum shapefile encontrado para processar.")
		return pd.DataFrame()

	orchestrator = DownloadOrchestrator(max_workers=max_workers)
	all_frames: List[pd.DataFrame] = []
	
	for target in targets:
		LOGGER.info("=== Processando Município: %s ===", target.shapefile_path.stem)
		codibge, _outputs, combined = orchestrator.process_municipio(
			target,
			start=start_date,
			end=end_date,
			output_dir=output_dir_path,
			cache_dir=cache_dir_path,
			disease=disease,
		)
		
		# Schema application per municipality
		if final_schema_norm == "inmet":
			combined = apply_output_schema(combined, INMET_OUTPUT_COLUMNS)
		elif final_schema_norm == "reference" and schema_cols is not None:
			combined = apply_output_schema(combined, schema_cols)

		if write_per_municipio:
			out_per_muni = output_dir_path / "final" / "by_municipio" / f"final_{codibge}.csv"
			out_per_muni.parent.mkdir(parents=True, exist_ok=True)
			combined.to_csv(out_per_muni, index=False)
			LOGGER.info(f"CSV final do município gerado -> {out_per_muni}")
			
		all_frames.append(combined)

	# Global Merge
	df_all = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
	if df_all.empty:
		LOGGER.warning("Nenhum dado foi gerado.")
		return df_all

	# Groupby final to remove duplicates
	key = ["codigo_ibge", "date"]
	df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce").dt.normalize()
	df_all = df_all.dropna(subset=["date"]).reset_index(drop=True)
	
	numeric_cols = [c for c in df_all.columns if c not in key and pd.api.types.is_numeric_dtype(df_all[c])]
	agg = {c: "mean" for c in numeric_cols}
	
	# Only aggregate if there are duplicates and key columns exist
	key_present = [k for k in key if k in df_all.columns]
	if len(key_present) == len(key):
		if df_all.duplicated(subset=key).any():
			df_all = df_all.groupby(key, as_index=False).agg(agg).sort_values(key).reset_index(drop=True)
	else:
		LOGGER.warning(f"Colunas chave {key} ausentes. Pulasdo agregação de duplicatas. Colunas presentes: {list(df_all.columns)}")
	
	# Final Schema Check
	if final_schema_norm == "reference" and schema_cols is not None:
		# Note: schema_cols might not have municipio/uf if reference didn't have them.
		df_all = apply_output_schema(df_all, schema_cols)

	# Enrich with Metadata (Municipio, UF) - Done AFTER schema to ensure they are not dropped
	try:
		ibge_path = HERE.parent.parent / "data/utils/municipios_ibge.csv" # Adjusted path relative to src/baixar_dados
		if not ibge_path.exists():
             # Fallback: try absolute or other relative
			ibge_path = Path("data/utils/municipios_ibge.csv")
            
		if ibge_path.exists():
			meta = pd.read_csv(ibge_path, dtype={"codigo_ibge": str})
			# Ensure 7 digits
			meta["codigo_ibge"] = meta["codigo_ibge"].str.zfill(7)
			
            # Ensure df_all has codigo_ibge (apply_output_schema might have renamed/preserved it)
			if "codigo_ibge" in df_all.columns:
				df_all["codigo_ibge"] = df_all["codigo_ibge"].astype(str).str.zfill(7)
				df_all = df_all.merge(meta[["codigo_ibge", "Nome_Município", "Nome_UF"]], on="codigo_ibge", how="left")
				df_all = df_all.rename(columns={"Nome_Município": "municipio", "Nome_UF": "uf"})
				
				# Reorder: codigo_ibge, municipio, uf, date, ... rest
				output_cols = ["codigo_ibge", "municipio", "uf", "date"]
				# Handle case if date/codibge were missing or diff named
				final_cols = []
				for c in output_cols:
					if c in df_all.columns:
						final_cols.append(c)
				
				rest = [c for c in df_all.columns if c not in final_cols]
				df_all = df_all[final_cols + rest]
			else:
				LOGGER.warning("Coluna 'codigo_ibge' perdida após aplicação do schema. Não foi possível enriquecer metadados.")

		else:
			LOGGER.warning(f"Metadados IBGE não encontrados em {ibge_path}. Colunas municipio/uf não serão adicionadas.")
	except Exception as e:
		LOGGER.warning(f"Erro ao enriquecer com metadados: {e}")

	if write_final:
		final_csv_path.parent.mkdir(parents=True, exist_ok=True)
		df_all.to_csv(final_csv_path, index=False)
		
		# Summary Report
		n_munis = df_all["codigo_ibge"].nunique() if "codigo_ibge" in df_all.columns else 0
		n_rows = len(df_all)
		LOGGER.info("=" * 40)
		LOGGER.info(f"RELATÓRIO FINAL:")
		LOGGER.info(f"Municípios processados: {n_munis}")
		LOGGER.info(f"Total de linhas geradas: {n_rows}")
		LOGGER.info(f"Arquivo final unificado: {final_csv_path}")
		LOGGER.info("=" * 40)

	return df_all