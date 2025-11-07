import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

sns.set(style="whitegrid")


def find_input_file():
    candidates = [
        "obesity_dataset_clean.csv",
        "obesity_hw_imputed.csv",
        "Tema_6.csv",
        "ObesityDataSet_raw_and_data_sinthetic.csv",
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
    # suppress non-critical warnings to keep terminal output clean
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
    if target:
        counts = df[target].value_counts()
        print('\n=== Distribución de ' + target + ' ===')
        pct = (counts / counts.sum() * 100).round(2)
        display_df = pd.DataFrame({'cantidad': counts, 'porcentaje': pct})
        print(display_df.to_string())
        n_classes = len(counts)
        palette = sns.color_palette('tab20', n_colors=n_classes)
        plt.figure(figsize=(8,6))
        sns.countplot(y=target, data=df, order=counts.index, palette=palette)
        plt.title('Distribución de NObeyesdad')
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
            group_stats = df.groupby(target)[col].agg(['count','mean','median','std']).sort_values('median')
            print(f"\n=== Estadísticas de {col} por {target} ===")
            gs = group_stats.rename(columns={'count':'cantidad','mean':'media','median':'mediana','std':'desviación_estándar'})
            print(gs.round(3).to_string())

            plt.figure(figsize=(10,6))
            order = group_stats.sort_values('median').index
            sns.boxplot(x=col, y=target, data=df, order=order, palette='vlag')
            plt.title(f'{col} por {target}')
            plt.tight_layout()
            plt.savefig(f'outputs/box_{col}_by_{target}.png', dpi=150)
            plt.close()

    # Height vs Weight scatter 
    if 'Height' in df.columns and 'Weight' in df.columns:
        sub = df.sample(frac=0.25, random_state=42) if len(df) > 2000 else df
        pair = df[['Height','Weight']].dropna()
        corr = pair.corr().iloc[0,1]
        print(f"\nAltura vs Peso: pares_n={len(pair)}, correlación_pearson={corr:.3f}")

        plt.figure(figsize=(8,6))
        if target:
            palette = sns.color_palette('tab20', n_colors=df[target].nunique())
            sns.scatterplot(x='Height', y='Weight', hue=target, data=sub, palette=palette, alpha=0.9, s=40)
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
            cros = pd.crosstab(df[col], df[target], normalize='index')*100
            top = cros.iloc[:20]
            print(f"\n=== Tabla % de {col} por {target} (primeras 20 filas) ===")
            print(top.round(2).to_string())
            n_series = top.shape[1]
            colors = sns.color_palette('tab20', n_colors=n_series)
            ax = top.plot(kind='bar', stacked=True, figsize=(10,6), color=colors)
            plt.ylabel('% within category')
            plt.title(f'{col} (%) by {target}')
            plt.tight_layout()
            outname = f'outputs/{col}_by_{target}_stacked.png'
            plt.savefig(outname, dpi=150)
            plt.close()

    print('\nAll visualizations saved to outputs')


if __name__ == '__main__':
    main()
