"""Визуализация разбиений train/test и train/valid/test для проекта ML3."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

# Названия цветов берём отсюда:
# https://en.wikipedia.org/wiki/List_of_Crayola_crayon_colors
crayon_colors = [
    "Jungle Green",
    "Macaroni and Cheese",  # кремово-оранжевый
    "Fuzzy Wuzzy",  # коричнево-розовый
    "Blue Bell",  # сиренево-синий
    "Sea Green",  # морской зелёный
    "Sunglow",  # солнечно-жёлтый
    "Orchid",  # орхидея
    "Mango Tango",
    "Jazzberry Jam",  # малиновый
    "Asparagus",  # спаржевый зелёный
    "Orange", # #FF7538
    "Apricot"
    ]
palette = sns.crayon_palette(crayon_colors)

sns.set_theme(style="darkgrid", palette=palette)
sns.palplot(palette)
plt.show()

# Палитра цветов Crayola (классические яркие оттенки)
CRAYOLA = {
    "blue": "#1F75FE",
    "orange": "#FF7538",
    "green": "#1CAC78",
    "red": "#EE204D",
    "purple": "#7851A9",
    "yellow": "#FCE883",
    }

# Фиксированное сопоставление цветов для выборок
SET_COLORS = {
    "Train": CRAYOLA["blue"],
    "Valid": CRAYOLA["orange"],
    "Test": CRAYOLA["green"],
    }


def visualize_train_test_split(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """Строит boxplot и scatterplot для разбиения df_train/df_test."""
    df_train_tmp = df_train[["price"]].copy()
    df_train_tmp["set"] = "Train"
    
    df_test_tmp = df_test[["price"]].copy()
    df_test_tmp["set"] = "Test"
    
    combined = pd.concat([df_train_tmp, df_test_tmp])
    
    # sns.set_style("dark", {"axes.facecolor": sns.crayons["Periwinkle"]})
    
    # Timberwolf «Серый волк» теплый светло-серый или светло-серый с бежевым оттенком (поскольку цвет имитирует окраску шерсти этого животного).
    # timberwolf_hex = "#D9D6CF"
    sns.set_style("dark", {"axes.facecolor": sns.crayons["Timberwolf"]})
    palette1=palette
    
    # 1. Boxplot
    plt.figure(figsize=(12, 4))
    box_palette = sns.crayon_palette(crayon_colors[:2])
    sns.boxplot(
        x="set", y="price", data=combined, showmeans=True,
        palette=box_palette,
        # color=sns.crayon_palette(colors)
        hue="set"
        )
    plt.xlabel("")
    plt.tight_layout()
    plt.show()
    
    print(
        "Размер df_train = ", df_train.shape,
        "\t" * 18,
        "Размер df_test =", df_test.shape,
        )
    
    # 2. Scatterplot
    plt.figure(figsize=(12, 5))
    sns.scatterplot(
        data=df_train, y="price", x="created",
        palette=palette1,
        # color=SET_COLORS["Train"],
        alpha=0.8, label="Train"
        )
    sns.scatterplot(
        data=df_test, y="price", x="created",
        palette=palette1,
        # color=SET_COLORS["Test"],
        alpha=0.6, label="Test"
        )
    plt.legend(bbox_to_anchor=(1.038, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()


def visualize_train_test_split_old(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    ) -> None:
    """Строит boxplot и scatterplot для разбиения df_train/df_test."""
    df_train_tmp = df_train[["price"]].copy()
    df_train_tmp["set"] = "Train"

    df_test_tmp = df_test[["price"]].copy()
    df_test_tmp["set"] = "Test"

    combined = pd.concat([df_train_tmp, df_test_tmp])

    # 1. Boxplot
    sns.set_style("dark")
    plt.figure(figsize=(12, 4))
    sns.boxplot(
        x="set", y="price", data=combined, showmeans=True,
        palette={"Train": SET_COLORS["Train"], "Test": SET_COLORS["Test"]}
        )
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

    print(
        "Размер df_train = ", df_train.shape,
        "\t" * 18,
        "Размер df_test =", df_test.shape,
        )

    # 2. Scatterplot
    plt.figure(figsize=(12, 5))
    sns.scatterplot(
        data=df_train, y="price", x="created",
        color=SET_COLORS["Train"], alpha=0.8, label="Train"
        )
    sns.scatterplot(
        data=df_test, y="price", x="created",
        color=SET_COLORS["Test"], alpha=0.6, label="Test"
        )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()


def visualize_train_valid_test_split(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    oy:str,
    ox:str
    ) -> None:
    """Строит boxplot и scatterplot для разбиения df_train/df_valid/df_test."""
    df_train_tmp = df_train[[oy]].copy()
    df_train_tmp["set"] = "Train"
    
    df_valid_tmp = df_valid[[oy]].copy()
    df_valid_tmp["set"] = "Valid"
    
    df_test_tmp = df_test[[oy]].copy()
    df_test_tmp["set"] = "Test"
    
    combined = pd.concat([df_train_tmp, df_valid_tmp, df_test_tmp])
    
    sns.set_style("dark", {"axes.facecolor": sns.crayons["Timberwolf"]})
    
    palette1 = palette
    palette1 = []
    palette1.append(palette[0]) # цвет для трэйна
    palette1.append(palette[2]) # цвет для валидационной выборки
    palette1.append(palette[11]) # цвет для тестовой выборки
    
    
    # 1. Boxplot
    plt.figure(figsize=(10, 4))
    # sns.boxplot(
    #     x="set", y=oy, data=combined, showmeans=True,
    #     palette=palette1
    #     )

    sns.boxplot(
        x="set",
        y=oy,
        data=combined,
        showmeans=True,
        hue="set",          # Добавляем hue
        palette=palette1,
        legend=False        # Отключаем легенду, так как она уже есть по оси X
    )

    plt.xlabel("")
    plt.tight_layout()
    plt.show()
    
    print(
        "Размер df_train = ", df_train.shape,
        "\t" * 8,
        "Размер df_valid =", df_valid.shape,
        "\t" * 8,
        "Размер df_test =", df_test.shape,
        )
    
    # 2. Scatterplot
    plt.figure(figsize=(12, 5))
    sns.scatterplot(
        data=df_train, y=oy, x=ox,
        color=palette1[0],  # Явно указываем цвет
        alpha=0.8, label="Train"
        )
    sns.scatterplot(
        data=df_valid, y=oy, x=ox,
        color=palette1[1],
        alpha=0.8, label="Valid"
        )
    sns.scatterplot(
        data=df_test, y=oy, x=ox,
        color=palette1[2],
        alpha=0.6, label="Test"
        )
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Освобождаем место справа для легенды
    fig = plt.gcf()
    fig.subplots_adjust(right=0.85)
    plt.show()





def visualize_train_valid_test_split_old(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    ) -> None:
    """Строит boxplot и scatterplot для разбиения df_train/df_valid/df_test."""
    df_train_tmp = df_train[["price"]].copy()
    df_train_tmp["set"] = "Train"
    
    df_valid_tmp = df_valid[["price"]].copy()
    df_valid_tmp["set"] = "Valid"
    
    df_test_tmp = df_test[["price"]].copy()
    df_test_tmp["set"] = "Test"
    
    combined = pd.concat([df_train_tmp, df_valid_tmp, df_test_tmp])
    
    # 1. Boxplot
    sns.set_style("dark")
    plt.figure(figsize=(10, 4))
    sns.boxplot(
        x="set", y="price", data=combined, showmeans=True,
        palette={
            "Train": SET_COLORS["Train"],
            "Valid": SET_COLORS["Valid"],
            "Test": SET_COLORS["Test"],
            }
        )
    plt.xlabel("")
    plt.tight_layout()
    plt.show()
    
    print(
        "Размер df_train = ", df_train.shape,
        "\t" * 8,
        "Размер df_valid =", df_valid.shape,
        "\t" * 8,
        "Размер df_test =", df_test.shape,
        )
    
    # 2. Scatterplot
    plt.figure(figsize=(12, 5))
    sns.scatterplot(
        data=df_train, y="price", x="created",
        color=SET_COLORS["Train"], alpha=0.8, label="Train"
        )
    sns.scatterplot(
        data=df_valid, y="price", x="created",
        color=SET_COLORS["Valid"], alpha=0.8, label="Valid"
        )
    sns.scatterplot(
        data=df_test, y="price", x="created",
        color=SET_COLORS["Test"], alpha=0.6, label="Test"
        )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()


def visualize_k_fold_split(
    df: pd.DataFrame,
    splits: list[tuple],
    ) -> None:
    """Строит heatmap индексов и boxplot price для K-Fold разбиения."""
    sns.set_style("dark", {"axes.facecolor": sns.crayons["Timberwolf"]})
    # ---------- Данные для heatmap ----------
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        rows.extend([{'fold': fold, 'index': idx, 'set': 'Train'} for idx in train_idx])
        rows.extend([{'fold': fold, 'index': idx, 'set': 'Test'} for idx in test_idx])
    
    heatmap_df = pd.DataFrame(rows)
    heatmap_df['value'] = heatmap_df['set'].map({'Train': 0, 'Test': 1})
    heatmap_matrix = heatmap_df.pivot(index='fold', columns='index', values='value')
    
    # ---------- Данные для boxplot ----------
    boxplot_parts = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        df_train = df.iloc[train_idx][['price']].copy()
        df_train['fold'] = fold
        df_train['set'] = 'Train'
        
        df_test = df.iloc[test_idx][['price']].copy()
        df_test['fold'] = fold
        df_test['set'] = 'Test'
        
        boxplot_parts.append(df_train)
        boxplot_parts.append(df_test)
    
    boxplot_df = pd.concat(boxplot_parts)
    
    # ---------- Общее полотно ----------
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 9), gridspec_kw={'height_ratios': [2, 1]})
    
    # График 1: heatmap в цветах Crayola
    heatmap_cmap = ListedColormap([palette[0], palette[11]])
    sns.heatmap(heatmap_matrix, cmap=heatmap_cmap, cbar=False, ax=axes[0])
    
    axes[0].set_title('K-Fold: Train/Test разбиение по индексам')
    axes[0].set_xlabel('Индекс строки')
    axes[0].set_ylabel('Fold')
    
    # График 2: boxplot в цветах Crayola
    sns.boxplot(
        data=boxplot_df, x='fold', y='price', hue='set', showmeans=True, ax=axes[1],
        palette=palette
        )
    
    axes[1].set_title('Распределение price в Train/Test по фолдам')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('price')
    axes[1].legend(title="Set", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
