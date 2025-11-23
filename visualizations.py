import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

sns.set(style="whitegrid")


def find_input_file():
    candidates = [
        "data_clean.csv",
        "Tema_6.csv",
    ]
    for f in candidates:
        if os.path.exists(f):
            return f
    raise FileNotFoundError("No input CSV found. Expected one of: " + ",".join(candidates))


def ensure_output_dirs():
    os.makedirs("outputs", exist_ok=True)


def main():
    infile = find_input_file()
    print("Loading:", infile)
    warnings.filterwarnings("ignore")
    sep = ";" if infile.endswith("_clean.csv") or infile.endswith("_hw_imputed.csv") or infile == "Tema_6.csv" else ","
    df = pd.read_csv(infile, sep=sep, skipinitialspace=True)

    ensure_output_dirs()

    if 'Weight' in df.columns and 'Height' in df.columns:
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if 'index' in num_cols:
        num_cols.remove('index')

    target = 'NObeyesdad' if 'NObeyesdad' in df.columns else None
    grouped_target = None
    
    FIXED_ORDER = ['Insufficient weight', 'Normal weight', 'Overweight', 'Obesity']
    if target:
        # Map 4 clases: Insufficient, Normal, Obesity, Overweight
        def map_to_four(c):
            if pd.isna(c):
                return np.nan
            s = str(c).strip().lower()
            if 'insufficient' in s or 'underweight' in s:
                return 'Insufficient weight'
            if s.startswith('normal') or s == 'normal':
                return 'Normal weight'
            if 'overweight' in s:
                return 'Overweight'
            if 'obesity' in s or 'obesity_type' in s or 'obesitytype' in s:
                return 'Obesity'
    
            return ' '.join([w.capitalize() for w in s.split()])

        grouped_col = target + '_grouped'
        df[grouped_col] = df[target].apply(map_to_four)
        df[grouped_col] = pd.Categorical(df[grouped_col], categories=FIXED_ORDER, ordered=True)
        grouped_target = grouped_col

        counts = df[grouped_target].value_counts().reindex(FIXED_ORDER).fillna(0)
        print('\n=== Distribución unificada (' + grouped_target + ') ===')
        pct = (counts / counts.sum() * 100).round(2)
        display_df = pd.DataFrame({'cantidad': counts, 'porcentaje': pct})
        print(display_df.to_string())

        available_order = [o for o in FIXED_ORDER if o in df[grouped_target].cat.categories and (df[grouped_target] == o).any()]
        palette = sns.color_palette('Set2', n_colors=len(available_order))
        plt.figure(figsize=(8,6))
        sns.countplot(y=grouped_target, data=df, order=available_order, palette=palette)
        plt.title('Distribución (agrupada) de NObeyesdad')
        plt.tight_layout()
        plt.savefig('outputs/class_distribution.png', dpi=150)
        plt.close()

    # Histogramas para algunas columnas numericas
    def print_numeric_summary(df, col):
        s = df[col].dropna()
        if s.empty:
            print(f'No hay datos en {col}')
            return
        desc = s.describe()
        iqr = desc['75%'] - desc['25%']
        lower = desc['25%'] - 1.5 * iqr
        upper = desc['75%'] + 1.5 * iqr
        outliers = s[(s < lower) | (s > upper)].shape[0]
        print(f"\n--- Estadísticas para {col} ---")
        print(f"n={int(desc['count'])}, media={desc['mean']:.3f}, mediana={desc['50%']:.3f}, desviación_estándar={desc['std']:.3f}")
        print(f"mín={desc['min']:.3f}, 25%={desc['25%']:.3f}, 75%={desc['75%']:.3f}, máx={desc['max']:.3f}")
        print(f"IQR={iqr:.3f}, outliers_regla_IQR={outliers}")

    for col in ['Age', 'Height', 'Weight', 'BMI']:
        if col in df.columns:
            print_numeric_summary(df, col)
            plt.figure(figsize=(6,4))
            sns.histplot(df[col].dropna(), bins=40, kde=True, color='#2b8cbe')
            plt.title(f'Distribución: {col}')
            plt.xlabel(col)
            plt.tight_layout()
            plt.savefig(f'outputs/hist_{col}.png', dpi=150)
            plt.close()

    # Boxplot de Age
    if target:
        col = 'Age'
        if col in df.columns:
            group_for_stats = grouped_target if grouped_target is not None else target
            if group_for_stats == grouped_target:
                group_stats = df.groupby(group_for_stats)[col].agg(['count','mean','median','std']).reindex(FIXED_ORDER).dropna(how='all').sort_values('median')
            else:
                group_stats = df.groupby(group_for_stats)[col].agg(['count','mean','median','std']).sort_values('median')
            print(f"\n=== Estadísticas de {col} por {group_for_stats} ===")
            gs = group_stats.rename(columns={'count':'cantidad','mean':'media','median':'mediana','std':'desviación_estándar'})
            print(gs.round(3).to_string())

            plt.figure(figsize=(10,6))
            order = [o for o in FIXED_ORDER if o in group_stats.index]
            sns.boxplot(x=col, y=group_for_stats, data=df, order=order, palette='vlag')
            plt.title(f'{col} por {group_for_stats}')
            plt.tight_layout()
            plt.savefig(f'outputs/box_{col}by{group_for_stats}.png', dpi=150)
            plt.close()

    # Height vs Weight scatter 
    if 'Height' in df.columns and 'Weight' in df.columns:
        sub = df.sample(frac=0.25, random_state=42) if len(df) > 2000 else df
        pair = df[['Height','Weight']].dropna()
        corr = pair.corr().iloc[0,1]
        print(f"\nAltura vs Peso: pares_n={len(pair)}, correlación_pearson={corr:.3f}")

        plt.figure(figsize=(8,6))
        if target:
            hue_col = grouped_target if grouped_target is not None else target
            palette = sns.color_palette('tab20', n_colors=len(FIXED_ORDER))
            sns.scatterplot(x='Height', y='Weight', hue=hue_col, data=sub, palette=palette, alpha=0.9, s=40)
            plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
        else:
            plt.scatter(sub['Height'], sub['Weight'], s=10, alpha=0.9, color='#1f78b4')
        plt.title('Height vs Weight')
        plt.tight_layout()
        plt.savefig('outputs/height_weight_scatter.png', dpi=150)
        plt.close()

    cat_cols = [c for c in ['MTRANS','FAVC','CAEC','CALC','SMOKE','SCC','family_history_with_overweight'] if c in df.columns]
    for col in cat_cols:
        if target:
            cross_col = grouped_target if grouped_target is not None else target
            cros = pd.crosstab(df[col], df[cross_col], normalize='index')*100
            present_cols = [c for c in FIXED_ORDER if c in cros.columns]
            cros = cros.reindex(columns=present_cols)
            top = cros.iloc[:20]
            print(f"\n=== Tabla % de {col} por {cross_col} (primeras 20 filas) ===")
            print(top.round(2).to_string())
            n_series = top.shape[1]
            colors = sns.color_palette('tab20', n_colors=n_series)
            ax = top.plot(kind='bar', stacked=True, figsize=(10,6), color=colors)
            plt.ylabel('% within category')
            plt.title(f'{col} (%) by {cross_col}')
            plt.tight_layout()
            outname = f'outputs/{col}by{cross_col}.png'
            plt.savefig(outname, dpi=150)
            plt.close()

    print('\nAll visualizations saved to outputs')


if __name__ == '__main__':
    main()