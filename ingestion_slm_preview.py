import pandas as pd
import os

# --- CONFIG ---
EXCEL_PATH = "uploads/Energy Consumption Daily Report MHS Ele - Copy.xlsx"

def is_date_col(col):
    """Check if column is a date (timestamp or string)"""
    from datetime import datetime
    if isinstance(col, pd.Timestamp):
        return True
    if isinstance(col, str):
        for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                datetime.strptime(col[:10], fmt)
                return True
            except Exception:
                pass
    # Also check for datetime objects
    if hasattr(col, 'year') and hasattr(col, 'month') and hasattr(col, 'day'):
        return True
    return False

def find_time_series_sheet(xl):
    """Find a typical monthly time series sheet"""
    for sheet, df in xl.items():
        date_cols = [col for col in df.columns if is_date_col(col)]
        if ("SWB NO." in df.columns or "SWB NO" in df.columns) and date_cols:
            return sheet, df, date_cols
    raise RuntimeError("No suitable time series sheet found!")

def preview_rows():
    xl = pd.read_excel(EXCEL_PATH, sheet_name=None)
    sheet, df, date_cols = find_time_series_sheet(xl)

    print(f"Previewing SLM row ingestion for sheet: {sheet}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Date columns found: {len(date_cols)}")
    
    # The first column (unnamed) contains feeder names
    fe_col = df.columns[0]  # First column is the feeder name
    swb_col = "SWB NO." if "SWB NO." in df.columns else "SWB NO"

    diff_map = {}
    # Try to map date cols to their difference columns (assuming sequence: date, diff, date, diff, ...)
    idxs = [df.columns.get_loc(dc) for dc in date_cols]
    columns = list(df.columns)
    for idx, dc in zip(idxs, date_cols):
        # Look one col right for corresponding difference col
        if idx + 1 < len(df.columns) and "Difference" in str(columns[idx+1]):
            diff_map[dc] = columns[idx+1]
        else:
            diff_map[dc] = None

    # For first 3 non-empty Feeder rows and first 5 dates
    eq_count = 0
    for ix, row in df.iterrows():
        fe = row.get(fe_col, "")
        swb = row.get(swb_col, "")
        if pd.isna(fe) or not str(fe).strip() or str(fe).strip() == "":
            continue
        printed = 0
        print(f"\n------ {fe} (SWB {swb}) ------")
        for dc in date_cols[:5]:
            reading = row.get(dc, None)
            if pd.isna(reading): continue
            diff = None
            if diff_map[dc]:
                diff = row.get(diff_map[dc], None)
            date_str = dc.strftime('%Y-%m-%d') if hasattr(dc, 'strftime') else str(dc)
            msg = f"feeder {fe} location {swb} on Date {date_str} {reading} KWH"
            if diff and not pd.isna(diff):
                msg += f", Difference vs previous day: {diff} KWH"
            print(msg)
            printed += 1
        eq_count += 1
        if eq_count>=3: break

if __name__ == "__main__":
    preview_rows()
    print("""
--- SLM Ingestion Prompt Example ---

Given a table row like:
| SWB NO. | Feeder | 2024-07-01 | Difference | 2024-07-02 | Difference | ...
| I/C-1   | SCP M/C FDR-2 | 246740 | 145 | 246885 | 138 | ...

Generate for each date column, a row as:
feeder <Feeder> location <SWB NO.> on Date <DATE> <Reading> KWH, Difference vs previous day: <Difference> KWH

Do this for all dates, one entry per line. Omit where the reading is blank.
""")