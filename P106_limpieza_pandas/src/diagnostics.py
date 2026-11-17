from pathlib import Path
import pandas as pd

ACTIVITY_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = ACTIVITY_DIR / "data" / "ventas.csv"
OUTPUT_FILE = ACTIVITY_DIR / "submission" / "ventas.csv"


def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def inspect_overview(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    print_section("1. RESUMEN GENERAL Y NULOS (ANTES VS. DESPUÉS)")
    print(f"Shape crudo : {df_raw.shape}")
    print(f"Shape limpio: {df_clean.shape}")
    print(f"Filas duplicadas en limpio: {df_clean.duplicated().sum()}")

    summary = pd.DataFrame({
        "dtype_limpio": df_clean.dtypes,
        "nulos_limpio": df_clean.isna().sum(),
        "unicos_limpio": df_clean.nunique(dropna=True),
    })
    print("\nDetalle por columna:")
    print(summary)


def inspect_categoricals(df_clean: pd.DataFrame):
    print_section("2. AUDITORÍA DE TEXTO Y CATEGORÍAS")
    for col in ["supplier", "country", "city"]:
        if col not in df_clean.columns:
            continue
        print(f"\n--- {col} (valores únicos) ---")
        values = sorted(df_clean[col].dropna().unique(), key=str)
        for val in values:
            print(f"  {repr(val)}")


def inspect_date_patterns(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    print_section("3. PATRONES DE FECHA (purchase_date)")

    if "purchase_date" in df_raw.columns:
        raw_patterns = (
            df_raw["purchase_date"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.replace(r"\d", "d", regex=True)
            .value_counts()
        )
        print("Patrones detectados en INPUT:")
        print(raw_patterns.to_string())

    if "purchase_date" in df_clean.columns:
        clean_patterns = (
            df_clean["purchase_date"]
            .dropna()
            .astype(str)
            .str.replace(r"\d", "d", regex=True)
            .value_counts()
        )
        print("\nPatrones detectados en OUTPUT (debe ser solo 'dddd-dd-dd'):")
        print(clean_patterns.to_string())

        # Muestra de valores extremos ordenados por longitud y fecha
        dates = sorted(df_clean["purchase_date"].dropna().unique())
        dates = sorted(dates, key=len)
        print(f"\nMínima fecha: {dates[0] if dates else None} | Máxima fecha: {dates[-1] if dates else None}")


def inspect_numerics(df_clean: pd.DataFrame):
    print_section("4. ESTADÍSTICAS NUMÉRICAS")
    num_cols = [c for c in ["amount", "discount", "weight", "unit_price"] if c in df_clean.columns]
    if num_cols:
        print(df_clean[num_cols].describe().T[["count", "min", "50%", "max"]])


def run_assertions(df_clean: pd.DataFrame):
    print_section("5. VALIDACIONES DE CALIDAD (ASSERTS)")

    # 1. Nombres de columnas normalizados
    assert all(col == col.strip().lower().replace(" ", "_") for col in df_clean.columns), (
        "Hay columnas sin normalizar."
    )

    # 2. País estandarizado
    if "country" in df_clean.columns:
        assert (df_clean["country"] == "COL").all(), "Existen valores distintos de 'COL' en country."

    # 3. Fechas en formato ISO estricto YYYY-MM-DD
    if "purchase_date" in df_clean.columns:
        non_null_dates = df_clean["purchase_date"].dropna().astype(str)
        assert non_null_dates.str.match(r"^\d{4}-\d{2}-\d{2}$").all(), (
            "Hay fechas que no cumplen YYYY-MM-DD."
        )
        pd.to_datetime(non_null_dates, format="%Y-%m-%d", errors="raise")

    # 4. Rangos numéricos válidos
    if "discount" in df_clean.columns:
        discounts = df_clean["discount"].dropna()
        assert ((discounts >= 0) & (discounts <= 1)).all(), "Hay descuentos fuera del rango [0, 1]."

    for col in ["amount", "weight", "unit_price"]:
        if col in df_clean.columns:
            vals = df_clean[col].dropna()
            assert (vals >= 0).all(), f"Hay valores negativos en {col}."

    print("Todas las validaciones pasaron correctamente.")


def main():
    df_raw = pd.read_csv(INPUT_FILE)
    df_clean = pd.read_csv(OUTPUT_FILE)

    inspect_overview(df_raw, df_clean)
    inspect_categoricals(df_clean)
    inspect_date_patterns(df_raw, df_clean)
    inspect_numerics(df_clean)
    run_assertions(df_clean)


if __name__ == "__main__":
    main()